# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

# core/safe_loader.py — 全版本兼容的模型安全加载器
# ═══════════════════════════════════════════════════════════════════
#
# 解决的问题:
#   1. transformers 版本不一致 → modeling_*.py 引用了不存在的内部函数
#   2. Unsloth 清理后 LazyAutoMapping 缓存损坏
#   3. pip 混合安装导致子模块版本不匹配
#   4. torch_dtype → dtype 参数重命名 (4.48+)
#   5. Windows 上 _LazyModule 行为不一致
#
# 设计原则:
#   - 不硬编码缺失函数列表 → 动态探测 + 自动修补
#   - 先试后补 → 只在真正失败时才注入 stub
#   - 不修改 transformers 源文件 → 纯运行时 patch
#   - 每个 stub 都是安全的 no-op → 不影响正常功能
#
# 用法:
#   from core.safe_loader import safe_load_model, ensure_model_importable
#   model = safe_load_model(model_path, trust_remote_code=True, ...)

from __future__ import annotations
import importlib
import importlib.util
import json
import logging
import re
import sys
import traceback
from pathlib import Path

_log = logging.getLogger("forgex.safe_loader")

# ═══════════════════════════════════════════════════════════════════
#  §1  版本检测
# ═══════════════════════════════════════════════════════════════════

_tf_version: tuple = (0, 0, 0)
_tf_root: Path | None = None

def _detect_transformers():
    global _tf_version, _tf_root
    try:
        import transformers
        _tf_root = Path(transformers.__file__).parent
        raw = getattr(transformers, "__version__", "0.0.0")
        nums = re.findall(r'\d+', raw)
        _tf_version = tuple(int(x) for x in nums[:3]) if nums else (0, 0, 0)
        return True
    except ImportError:
        return False

_HAS_TF = _detect_transformers()

# ═══════════════════════════════════════════════════════════════════
#  §2  兼容层 — 动态探测 + 修补缺失的内部函数
# ═══════════════════════════════════════════════════════════════════

# 已知的 stub 定义: (模块路径, 函数名, stub工厂)
# stub工厂 是一个 callable，返回合适的默认值
_KNOWN_STUBS = [
    # --- transformers.integrations (4.45+) ---
    ("transformers.integrations", "use_kernel_func_from_hub",  lambda: (lambda *a, **k: None)),
    ("transformers.integrations", "is_kernels_available",       lambda: (lambda: False)),
    ("transformers.integrations", "get_keys_to_not_convert",    lambda: (lambda model, **kw: [])),
    # --- transformers.utils (各版本) ---
    ("transformers.utils", "is_torch_npu_available",     lambda: (lambda: False)),
    ("transformers.utils", "is_torch_xla_available",     lambda: (lambda: False)),
    ("transformers.utils", "is_torch_mlu_available",     lambda: (lambda: False)),
    ("transformers.utils", "is_torch_musa_available",    lambda: (lambda: False)),
    ("transformers.utils", "is_torch_mps_available",     lambda: (lambda: False)),
    ("transformers.utils", "is_torch_sdpa_available",    lambda: (lambda: True)),
    ("transformers.utils", "is_flash_attn_2_available",  lambda: (lambda: False)),
    ("transformers.utils", "is_torchdynamo_available",   lambda: (lambda: False)),
    ("transformers.utils", "is_torchvision_available",   lambda: (lambda: False)),
    # --- transformers.modeling_utils (4.46+) ---
    ("transformers.modeling_utils", "ALL_ATTENTION_FUNCTIONS", lambda: {}),
    # --- transformers.processing_utils (4.46+) ---
    ("transformers.processing_utils", "Unpack", lambda: (lambda x: x)),
    # --- transformers.cache_utils (4.36+) ---
    ("transformers.cache_utils", "Cache",       lambda: type("Cache", (), {})),
    ("transformers.cache_utils", "DynamicCache", lambda: type("DynamicCache", (), {"__init__": lambda self: None})),
    ("transformers.cache_utils", "StaticCache",  lambda: type("StaticCache", (), {"__init__": lambda self, *a, **k: None})),
]

# 高阶 stubs: 装饰器、基类等，需要特殊处理
def _make_passthrough_decorator():
    """返回一个直通装饰器 (用于 can_return_tuple 等)"""
    def _deco(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda fn: fn
    return _deco

_DECORATOR_STUBS = [
    ("transformers.utils", "can_return_tuple", _make_passthrough_decorator),
    ("transformers.utils", "auto_docstring",   _make_passthrough_decorator),
]


def _inject(module_path: str, name: str, value) -> bool:
    """将 value 注入到 module_path 模块中（绕过 _LazyModule）。
    
    返回 True 如果确实注入了（原本不存在）。
    """
    try:
        mod = importlib.import_module(module_path)
    except Exception:
        return False

    # 检查是否已存在（用 __dict__ 直接查，避免触发 _LazyModule __getattr__）
    if name in mod.__dict__:
        return False
    # 再用 getattr 试一次（某些属性通过 __getattr__ 动态生成）
    try:
        getattr(mod, name)
        return False  # 已存在
    except (AttributeError, ImportError):
        pass

    # 注入
    mod.__dict__[name] = value
    # 同步到 sys.modules 缓存
    cached = sys.modules.get(module_path)
    if cached is not None and cached is not mod:
        cached.__dict__[name] = value
    return True


def _patch_transformers_compat():
    """修补 transformers 内部缺失的函数/类/属性。
    
    策略:
    1. 先扫描已知 stub 列表，只在确实缺失时注入
    2. 然后尝试 import 目标 modeling 模块，捕获 ImportError 做动态修补
    """
    if not _HAS_TF:
        return

    _patched = []

    # ── Phase 1: 已知 stubs ──
    for mod_path, func_name, factory in _KNOWN_STUBS:
        if _inject(mod_path, func_name, factory()):
            _patched.append(func_name)

    for mod_path, func_name, factory in _DECORATOR_STUBS:
        if _inject(mod_path, func_name, factory()):
            _patched.append(func_name)

    # ── Phase 2: 动态探测 — 尝试 import 常用 modeling 模块 ──
    # 如果失败，解析 ImportError 并自动注入缺失的名称
    _probe_targets = [
        "transformers.models.llama.modeling_llama",
        "transformers.models.mistral.modeling_mistral",
        "transformers.models.qwen2.modeling_qwen2",
    ]
    for target in _probe_targets:
        try:
            importlib.import_module(target)
        except ImportError as ie:
            fixed = _auto_fix_import_error(ie)
            if fixed:
                _patched.extend(fixed)
        except Exception:
            pass  # 非 ImportError 类型（如 CUDA 问题），不处理

    if _patched:
        _log.info(f"[safe_loader] 兼容层修补: {', '.join(dict.fromkeys(_patched))}")


def _auto_fix_import_error(err: ImportError) -> list:
    """解析 ImportError 消息，自动注入缺失的名称。
    
    处理两种常见格式:
      - "cannot import name 'xxx' from 'transformers.yyy'"
      - "No module named 'transformers.yyy.zzz'"
    """
    fixed = []
    msg = str(err)

    # 格式1: cannot import name 'xxx' from 'module.path'
    m = re.search(r"cannot import name '(\w+)' from '([\w.]+)'", msg)
    if m:
        name, mod_path = m.group(1), m.group(2)
        # 为缺失名称构造安全 stub
        stub = _make_safe_stub(name)
        if _inject(mod_path, name, stub):
            fixed.append(name)
            _log.info(f"[safe_loader] 动态修补: {mod_path}.{name}")
            return fixed

    # 格式2: No module named 'xxx.yyy.zzz'
    m = re.search(r"No module named '([\w.]+)'", msg)
    if m:
        missing_mod = m.group(1)
        # 创建一个空模块占位
        if missing_mod not in sys.modules:
            import types
            placeholder = types.ModuleType(missing_mod)
            placeholder.__path__ = []
            placeholder.__package__ = missing_mod.rsplit(".", 1)[0] if "." in missing_mod else missing_mod
            sys.modules[missing_mod] = placeholder
            fixed.append(f"module:{missing_mod}")
            _log.info(f"[safe_loader] 动态创建占位模块: {missing_mod}")

    return fixed


def _make_safe_stub(name: str):
    """根据名称推断并创建安全的 stub 值。"""
    lower = name.lower()

    # is_xxx_available → return False
    if lower.startswith("is_") and lower.endswith("_available"):
        return lambda: False

    # ALL_xxx / xxx_MAPPING → empty dict
    if lower.startswith("all_") or lower.endswith("_mapping") or lower.endswith("_functions"):
        return {}

    # XxxCache → empty class
    if lower.endswith("cache"):
        return type(name, (), {"__init__": lambda self, *a, **k: None})

    # xxx_decorator / auto_docstring / can_return_tuple → passthrough decorator
    if any(kw in lower for kw in ("decorator", "docstring", "return_tuple", "deprecat")):
        return _make_passthrough_decorator()

    # Unpack, unpack_xxx → identity
    if lower.startswith("unpack"):
        return lambda x: x

    # 默认: no-op 函数
    return lambda *a, **k: None


# ═══════════════════════════════════════════════════════════════════
#  §3  dtype 兼容: torch_dtype vs dtype
# ═══════════════════════════════════════════════════════════════════

_dtype_param_name: str | None = None  # 缓存结果

def dtype_kwarg(torch_dtype) -> dict:
    """返回正确的 dtype 关键字参数 (兼容 transformers 4.30 — 4.50+)。
    
    transformers >= 4.48 将 torch_dtype 重命名为 dtype。
    本函数自动探测并返回 {"torch_dtype": ...} 或 {"dtype": ...}。
    
    用法: mk.update(dtype_kwarg(torch.float16))
    """
    global _dtype_param_name
    if _dtype_param_name is None:
        _dtype_param_name = _detect_dtype_param()
    return {_dtype_param_name: torch_dtype}


def _detect_dtype_param() -> str:
    """探测 from_pretrained 接受 dtype 还是 torch_dtype"""
    try:
        import inspect
        from transformers import AutoModelForCausalLM
        sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
        params = sig.parameters
        if "dtype" in params and "torch_dtype" not in params:
            return "dtype"
        if "torch_dtype" in params:
            return "torch_dtype"
    except Exception:
        pass
    # 版本推断 fallback
    if _tf_version >= (4, 48, 0):
        return "dtype"
    return "torch_dtype"


# ═══════════════════════════════════════════════════════════════════
#  §4  架构类名 → 子模块路径 映射表
# ═══════════════════════════════════════════════════════════════════

_CLASS_TO_MODULE = {
    "LlamaForCausalLM":        "transformers.models.llama.modeling_llama",
    "MistralForCausalLM":      "transformers.models.mistral.modeling_mistral",
    "Qwen2ForCausalLM":        "transformers.models.qwen2.modeling_qwen2",
    "Qwen2MoeForCausalLM":     "transformers.models.qwen2_moe.modeling_qwen2_moe",
    "GemmaForCausalLM":        "transformers.models.gemma.modeling_gemma",
    "Gemma2ForCausalLM":       "transformers.models.gemma2.modeling_gemma2",
    "PhiForCausalLM":          "transformers.models.phi.modeling_phi",
    "Phi3ForCausalLM":         "transformers.models.phi3.modeling_phi3",
    "GPT2LMHeadModel":         "transformers.models.gpt2.modeling_gpt2",
    "GPTNeoXForCausalLM":      "transformers.models.gpt_neox.modeling_gpt_neox",
    "BloomForCausalLM":        "transformers.models.bloom.modeling_bloom",
    "OPTForCausalLM":          "transformers.models.opt.modeling_opt",
    "FalconForCausalLM":       "transformers.models.falcon.modeling_falcon",
    "Starcoder2ForCausalLM":   "transformers.models.starcoder2.modeling_starcoder2",
    "CodeGenForCausalLM":      "transformers.models.codegen.modeling_codegen",
    "CohereForCausalLM":       "transformers.models.cohere.modeling_cohere",
    "MixtralForCausalLM":      "transformers.models.mixtral.modeling_mixtral",
    "StableLmForCausalLM":     "transformers.models.stablelm.modeling_stablelm",
    "OlmoForCausalLM":         "transformers.models.olmo.modeling_olmo",
    "MptForCausalLM":          "transformers.models.mpt.modeling_mpt",
    "InternLMForCausalLM":     "transformers.models.llama.modeling_llama",  # Yi/InternLM often llama-based
    "DeepseekV2ForCausalLM":   "transformers.models.llama.modeling_llama",
}


# ═══════════════════════════════════════════════════════════════════
#  §5  config.json 读取
# ═══════════════════════════════════════════════════════════════════

def _read_architectures(model_path_or_id: str) -> list:
    """从 config.json 读取 architectures 字段。优先本地文件系统，fallback 到 AutoConfig。"""
    # 1) 本地 config.json
    p = Path(model_path_or_id)
    for cfg_path in [p / "config.json", p]:
        if cfg_path.is_file() and cfg_path.name == "config.json":
            try:
                return json.loads(cfg_path.read_text(encoding="utf-8")).get("architectures", [])
            except Exception:
                pass
    if p.is_dir():
        cfg = p / "config.json"
        if cfg.exists():
            try:
                return json.loads(cfg.read_text(encoding="utf-8")).get("architectures", [])
            except Exception:
                pass

    # 2) HuggingFace Hub / AutoConfig
    if _HAS_TF:
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_path_or_id, trust_remote_code=True)
            return getattr(cfg, "architectures", []) or []
        except Exception:
            pass
    return []


# ═══════════════════════════════════════════════════════════════════
#  §6  模型类解析 — 5 层递进策略
# ═══════════════════════════════════════════════════════════════════

# 已知会出现循环导入/半初始化问题的模块
_FRAGILE_MODULES = ["torchvision", "torchaudio", "torchtext", "cv2"]


def _cleanup_broken_modules():
    """清理 sys.modules 中半初始化（partially initialized）的模块。
    
    某些模块（尤其是 torchvision）在被其他库间接 import 时可能
    进入半初始化状态，后续任何碰到它们的 import 都会失败。
    
    解决方案: 检测并移除这些"僵尸模块"，让下一次 import 从头开始。
    """
    cleaned = []
    for prefix in _FRAGILE_MODULES:
        mod = sys.modules.get(prefix)
        if mod is None:
            continue
        # 检测半初始化: 模块对象存在但缺少关键属性
        is_broken = False
        try:
            spec = getattr(mod, "__spec__", None)
            if spec is not None and getattr(spec, "_initializing", False):
                is_broken = True
            if not is_broken and prefix == "torchvision":
                _ = mod.extension
        except (AttributeError, ImportError):
            is_broken = True
        except Exception:
            is_broken = True

        if is_broken:
            keys_to_remove = [k for k in sys.modules
                              if k == prefix or k.startswith(prefix + ".")]
            for k in keys_to_remove:
                del sys.modules[k]
            cleaned.append(prefix)

    if cleaned:
        _log.info(f"[safe_loader] 清除半初始化模块: {', '.join(cleaned)}")
    return cleaned


def _ensure_torchvision_safe():
    """确保 torchvision 不会炸掉 import 链。
    
    核心逻辑在 core/__init__.py 的 _torchvision_init()。
    此函数作为二次检查：如果 core 的 guard 没覆盖到，这里补救。
    """
    tv = sys.modules.get("torchvision")
    if tv is not None:
        try:
            _ = tv.extension._has_ops
            _ = tv.transforms.InterpolationMode
            return  # 健康
        except Exception:
            pass
        # 僵尸或半初始化 → 激活 Guardian mock 模式
        _log.warning("[safe_loader] torchvision 状态异常，激活 mock 模式")
        try:
            from core import _get_guardian, _clean_torchvision_modules, _build_torchvision_mock, _install_guardian
            guardian = _get_guardian()
            if guardian:
                guardian.activate_mock_mode()
            else:
                _clean_torchvision_modules()
                sys.modules.update(_build_torchvision_mock())
                _install_guardian()
            _log.info("[safe_loader] torchvision mock 已激活")
        except Exception as e:
            _log.error(f"[safe_loader] torchvision 修复失败: {e}")


def _resolve_class(class_name: str, retry_after_patch: bool = True):
    """尝试解析模型类。失败时自动修补环境后重试一次。"""
    # 首次调用前确保 torchvision 等不会炸
    _ensure_torchvision_safe()

    cls = _resolve_class_inner(class_name)
    if cls is not None:
        return cls

    # 所有策略都失败了 → 分析失败原因并针对性修复
    if retry_after_patch:
        mod_path = _CLASS_TO_MODULE.get(class_name)
        if mod_path:
            try:
                importlib.import_module(mod_path)
            except ImportError as ie:
                fixed = _auto_fix_import_error(ie)
                if fixed:
                    _log.info(f"[safe_loader] 动态修补后重试: {class_name}")
                    return _resolve_class(class_name, retry_after_patch=False)
            except Exception as e:
                err_msg = str(e).lower()
                if "partially initialized" in err_msg or "circular import" in err_msg:
                    _log.info(f"[safe_loader] 检测到半初始化模块冲突，激活 torchvision mock...")
                    try:
                        from core import _get_guardian
                        guardian = _get_guardian()
                        if guardian:
                            guardian.activate_mock_mode()
                    except Exception:
                        _ensure_torchvision_safe()
                    return _resolve_class(class_name, retry_after_patch=False)
    return None


def _resolve_class_inner(class_name: str):
    """6 层递进策略解析模型类。"""
    mod_path = _CLASS_TO_MODULE.get(class_name)

    # ══ 策略 0: 检查 sys.modules 缓存（最快）══
    if mod_path and mod_path in sys.modules:
        cls = getattr(sys.modules[mod_path], class_name, None)
        if cls is not None:
            return cls

    # ══ 策略 1: importlib.util.find_spec + exec_module（绕过 _LazyModule）══
    if mod_path:
        try:
            spec = importlib.util.find_spec(mod_path)
            if spec and spec.loader:
                parent_pkg = mod_path.rsplit(".", 1)[0]
                _ensure_parent_loaded(parent_pkg)
                mod = importlib.util.module_from_spec(spec)
                mod.__package__ = parent_pkg
                sys.modules[mod_path] = mod
                spec.loader.exec_module(mod)
                cls = getattr(mod, class_name, None)
                if cls is not None:
                    return cls
        except Exception as e:
            _log.warning(f"[resolve] 策略1({mod_path}): {_short_err(e)}")
            # 如果是半初始化问题，清理后让后续策略继续
            if "partially initialized" in str(e).lower():
                _cleanup_broken_modules()

    # ══ 策略 2: 从磁盘定位 .py 文件 + spec_from_file_location ══
    if mod_path and _tf_root:
        try:
            relative = mod_path.replace("transformers.", "").replace(".", "/") + ".py"
            py_file = _tf_root / relative
            if py_file.exists():
                parent_pkg = mod_path.rsplit(".", 1)[0]
                _ensure_parent_loaded(parent_pkg)
                spec = importlib.util.spec_from_file_location(
                    mod_path, py_file, submodule_search_locations=[])
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    mod.__package__ = parent_pkg
                    sys.modules[mod_path] = mod
                    spec.loader.exec_module(mod)
                    cls = getattr(mod, class_name, None)
                    if cls is not None:
                        return cls
        except Exception as e:
            _log.warning(f"[resolve] 策略2({mod_path}): {_short_err(e)}")
            if "partially initialized" in str(e).lower():
                _cleanup_broken_modules()

    # ══ 策略 3: 清缓存后 importlib.import_module ══
    if mod_path:
        try:
            sys.modules.pop(mod_path, None)
            parent = mod_path.rsplit(".", 1)[0] if "." in mod_path else None
            if parent and parent in sys.modules:
                pm = sys.modules[parent]
                if hasattr(pm, "_modules") and isinstance(pm._modules, dict):
                    pm._modules.pop(mod_path.split(".")[-1], None)
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, class_name, None)
            if cls is not None:
                return cls
        except Exception as e:
            _log.warning(f"[resolve] 策略3({mod_path}): {_short_err(e)}")
            if "partially initialized" in str(e).lower():
                _cleanup_broken_modules()

    # ══ 策略 4: transformers 顶层 getattr ══
    if _HAS_TF:
        try:
            import transformers
            cls = getattr(transformers, class_name, None)
            if cls is not None:
                return cls
        except Exception:
            pass

    # ══ 策略 5: 暴力文件搜索 ══
    if _tf_root:
        cls = _brute_force_search(class_name)
        if cls is not None:
            return cls

    return None


def _brute_force_search(class_name: str):
    """在 transformers/models/ 目录下搜索包含目标类定义的 .py 文件。"""
    models_dir = _tf_root / "models"
    if not models_dir.is_dir():
        return None

    # 优先搜索 class_name 推断的子目录
    lower = class_name.lower()
    for suffix in ("forcausallm", "lmheadmodel", "forsequenceclassification",
                    "formaskedlm", "forquestionanswering"):
        if lower.endswith(suffix):
            type_guess = lower[:-len(suffix)]
            break
    else:
        type_guess = lower

    # 名称归一化
    _aliases = {"gptneox": "gpt_neox", "qwen2moe": "qwen2_moe",
                "gpt2lm": "gpt2", "starcoder2": "starcoder2"}
    type_guess = _aliases.get(type_guess, type_guess)

    # 候选文件: 先搜推断目录，再搜全部
    candidates = []
    guess_file = models_dir / type_guess / f"modeling_{type_guess}.py"
    if guess_file.exists():
        candidates.append(guess_file)

    target_str = f"class {class_name}"
    for subdir in models_dir.iterdir():
        if not subdir.is_dir():
            continue
        for py in subdir.glob("modeling_*.py"):
            if py not in candidates:
                candidates.append(py)

    for candidate in candidates:
        try:
            text = candidate.read_text(encoding="utf-8", errors="ignore")
            if target_str not in text:
                continue
        except Exception:
            continue

        try:
            mod_name = f"transformers.models.{candidate.parent.name}.{candidate.stem}"
            parent_pkg = f"transformers.models.{candidate.parent.name}"
            _ensure_parent_loaded(parent_pkg)
            spec = importlib.util.spec_from_file_location(mod_name, candidate)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                mod.__package__ = parent_pkg
                sys.modules[mod_name] = mod
                spec.loader.exec_module(mod)
                cls = getattr(mod, class_name, None)
                if cls is not None:
                    _log.info(f"[resolve] 策略5(暴力搜索): {class_name} ← {candidate.name}")
                    return cls
        except Exception as e:
            _log.warning(f"[resolve] 策略5: {candidate.name}: {_short_err(e)}")
    return None


def _ensure_parent_loaded(parent_pkg: str):
    """确保父包已在 sys.modules 中（relative import 需要）。"""
    if parent_pkg in sys.modules:
        return
    parts = parent_pkg.split(".")
    for i in range(1, len(parts) + 1):
        sub = ".".join(parts[:i])
        if sub not in sys.modules:
            try:
                importlib.import_module(sub)
            except Exception:
                pass


def _short_err(e: Exception) -> str:
    """截断错误消息到一行。"""
    s = str(e).replace("\n", " ")
    return s[:120] + "..." if len(s) > 120 else s


# ═══════════════════════════════════════════════════════════════════
#  §7  公开 API
# ═══════════════════════════════════════════════════════════════════

def ensure_model_importable(model_path_or_id: str) -> None:
    """在调用 from_pretrained 之前预导入模型类。
    
    读取 config.json → architectures → 强制 import modeling 子模块。
    会自动修补环境中缺失的依赖。
    """
    _ensure_torchvision_safe()
    _patch_transformers_compat()

    archs = _read_architectures(model_path_or_id)
    for arch_name in archs:
        cls = _resolve_class(arch_name)
        if cls is not None:
            _log.info(f"[safe_loader] 预导入 {arch_name} ✓")
        else:
            _log.warning(f"[safe_loader] 预导入 {arch_name} ✗（自定义模型可能需要 trust_remote_code）")


def safe_load_model(model_path_or_id: str, **kwargs):
    """AutoModelForCausalLM.from_pretrained 的安全替代品。
    
    1. 执行兼容修补
    2. 深度诊断 import 链（捕获被 transformers 吞掉的真实错误）
    3. 调用 AutoModelForCausalLM.from_pretrained
    4. 如果失败 → 尝试 AutoModelForConditionalGeneration（多模态模型）
    5. 如果还失败 → 用手动解析的类直接加载
    """
    from transformers import AutoModelForCausalLM

    # Step 1+2: 修补 + 预导入
    ensure_model_importable(model_path_or_id)

    # Step 2.5: 深度 import 诊断
    # transformers 的 lazy import 会吞掉真实错误（numpy/scipy/sklearn 崩溃）
    # 变成无意义的 "Could not import module 'XXX'"
    # 在这里先直接 import 模型类，暴露真实错误
    archs = _read_architectures(model_path_or_id)
    _real_import_error = None
    for arch_name in archs:
        # 尝试直接 import modeling 模块
        _guessed_type = arch_name.lower()
        for suffix in ("forcausallm", "forsequenceclassification",
                        "forconditionalgeneration", "formaskedlm", "model"):
            if _guessed_type.endswith(suffix):
                _guessed_type = _guessed_type[:-len(suffix)]
                break
        try:
            _mod_name = f"transformers.models.{_guessed_type}.modeling_{_guessed_type}"
            importlib.import_module(_mod_name)
        except ImportError as ie:
            # 可能是 numpy/scipy/sklearn 崩溃
            ie_str = str(ie)
            if "numpy" in ie_str.lower() or "scipy" in ie_str.lower() or "sklearn" in ie_str.lower():
                _real_import_error = ie
                _log.error(
                    f"[safe_loader] ⚠️ 检测到 numpy/scipy 兼容性问题!\n"
                    f"  真实错误: {ie}\n"
                    f"  修复: pip install \"numpy<2.0\" scipy scikit-learn --force-reinstall"
                )
                # 尝试修复: 禁用 sklearn 在 generation 中的使用
                try:
                    sys.modules.setdefault("sklearn", type(sys)("sklearn"))
                    sys.modules.setdefault("sklearn.metrics", type(sys)("sklearn.metrics"))
                    _fake_sklearn = sys.modules["sklearn.metrics"]
                    if not hasattr(_fake_sklearn, "roc_curve"):
                        _fake_sklearn.roc_curve = None
                    _log.info("[safe_loader] 注入 sklearn shim（绕过 numpy 不兼容）")
                except Exception:
                    pass
            else:
                _real_import_error = ie
        except Exception as e:
            _real_import_error = e

    # Step 3: 正常加载（CausalLM）
    first_err = ""
    try:
        return AutoModelForCausalLM.from_pretrained(model_path_or_id, **kwargs)
    except Exception as e:
        first_err = str(e)
        # 只有映射/导入相关错误才继续 fallback
        _recoverable = ("Could not find", "Could not import", "ConditionalGeneration",
                        "not supported", "does not appear to have")
        if not any(kw in first_err for kw in _recoverable):
            raise

    # Step 4: 多模态模型 fallback
    _is_multimodal = any("ConditionalGeneration" in a or "ForVision" in a for a in archs)

    if _is_multimodal:
        _log.warning(f"[safe_loader] 检测到多模态模型架构 {archs}，尝试 AutoModelForConditionalGeneration...")
        try:
            from transformers import AutoModelForConditionalGeneration
            return AutoModelForConditionalGeneration.from_pretrained(model_path_or_id, **kwargs)
        except Exception as e2:
            _log.warning(f"[safe_loader] AutoModelForConditionalGeneration 也失败: {e2}")

        for arch_name in archs:
            causal_name = arch_name.replace("ForConditionalGeneration", "ForCausalLM")
            if causal_name != arch_name:
                cls = _resolve_class(causal_name)
                if cls is not None:
                    _log.info(f"[safe_loader] 多模态→纯文本骨干 fallback → {cls.__name__}")
                    try:
                        return cls.from_pretrained(model_path_or_id, **kwargs)
                    except Exception:
                        pass

    # Step 5: 通用 fallback — 直接类加载
    _log.warning("[safe_loader] AutoModel 映射失败，尝试直接类加载...")
    _patch_transformers_compat()

    diag_errors = []
    for arch_name in archs:
        cls = _resolve_class(arch_name)
        if cls is not None:
            _log.info(f"[safe_loader] fallback → {cls.__name__}.from_pretrained")
            return cls.from_pretrained(model_path_or_id, **kwargs)
        mod_path = _CLASS_TO_MODULE.get(arch_name, "")
        if mod_path:
            try:
                importlib.import_module(mod_path)
            except Exception as ie:
                diag_errors.append(f"{mod_path}: {ie}")

    # 构建诊断
    try:
        tf_ver = ".".join(str(x) for x in _tf_version)
    except Exception:
        tf_ver = "unknown"

    diag = (
        f"无法加载模型: {model_path_or_id}\n"
        f"架构: {archs}\n"
        f"transformers 版本: {tf_ver}\n"
        f"原始错误: {first_err}\n"
    )
    # ★ 显示被吞掉的真实错误
    if _real_import_error is not None:
        diag += f"\n⚠️ 底层真实错误（被 transformers 隐藏）:\n  {type(_real_import_error).__name__}: {_real_import_error}\n"
        if "numpy" in str(_real_import_error).lower():
            diag += (
                "\n🔧 这是 numpy 版本不兼容问题！请执行:\n"
                '  pip install "numpy<2.0" scipy scikit-learn --force-reinstall\n'
            )
    if _is_multimodal:
        diag += (
            "\n⚠️ 这是一个多模态模型（视觉+文本）。\n"
            "  如果只需要纯文本训练，建议使用同系列的纯文本版本。\n"
            "  例: google/gemma-3-27b-it → google/gemma-2-27b-it\n"
        )
    if diag_errors:
        diag += "导入诊断:\n" + "\n".join(f"  ✗ {e}" for e in diag_errors) + "\n"
    diag += (
        "\n建议修复方法:\n"
        "  1. pip install -U transformers  (推荐)\n"
        "  2. pip install transformers --force-reinstall  (完全重装)\n"
        "  3. 检查是否混合使用了 pip 和 conda 安装 transformers"
    )
    raise RuntimeError(diag)


def diagnose_environment() -> str:
    """输出环境诊断信息，帮助排查加载失败。"""
    lines = ["═══ ForgeX 环境诊断 ═══"]

    # transformers
    try:
        import transformers
        lines.append(f"transformers: {transformers.__version__} ({transformers.__file__})")
    except ImportError:
        lines.append("transformers: ❌ 未安装")

    # torch
    try:
        import torch
        cuda = f"CUDA {torch.version.cuda}" if torch.cuda.is_available() else "CPU only"
        lines.append(f"torch: {torch.__version__} ({cuda})")
    except ImportError:
        lines.append("torch: ❌ 未安装")

    # peft
    try:
        import peft
        lines.append(f"peft: {peft.__version__}")
    except ImportError:
        lines.append("peft: 未安装 (LoRA 不可用)")

    # bitsandbytes
    try:
        import bitsandbytes
        lines.append(f"bitsandbytes: {bitsandbytes.__version__}")
    except ImportError:
        lines.append("bitsandbytes: 未安装 (QLoRA 不可用)")

    # 测试关键 import
    test_imports = [
        ("transformers.models.llama.modeling_llama", "LlamaForCausalLM"),
        ("transformers.integrations", "use_kernel_func_from_hub"),
        ("transformers.cache_utils", "DynamicCache"),
    ]
    lines.append("\n关键导入测试:")
    for mod, attr in test_imports:
        try:
            m = importlib.import_module(mod)
            getattr(m, attr)
            lines.append(f"  ✓ {mod}.{attr}")
        except Exception as e:
            lines.append(f"  ✗ {mod}.{attr}: {e}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  §8  启动时执行
# ═══════════════════════════════════════════════════════════════════

# import safe_loader 时立即执行一次兼容修补
if _HAS_TF:
    _patch_transformers_compat()
