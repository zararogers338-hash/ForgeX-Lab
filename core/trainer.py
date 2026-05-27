# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

"""ForgeX trainer engine — v3.0 (全面质量优化版).

核心设计理念：让 1B 模型逼近 10B 效果
  - Completion-only loss（只训 response 部分 → 学习效率翻倍）
  - NEFTune 噪声嵌入（+25% MT-Bench，论文验证）
  - Cosine LR scheduler + warmup（平滑收敛，避免灾难性遗忘）
  - 全线性层 LoRA（不止 attention，还有 MLP → 捕获更多知识）
  - Chat template 对齐 + EOS token 对齐
  - FlashAttention 2 自动检测（内存减半+加速）
  - Gradient checkpointing（省 60% 激活内存）
  - Weight decay + gradient clipping（防过拟合+防梯度爆炸）
  - 自动验证集 + EarlyStopping

v3.0 新增:
  - DoRA（Weight-Decomposed LoRA，同 rank 下一致优于标准 LoRA）
  - rsLoRA（Rank-Stabilized scaling，高 rank 更稳定）
  - Label Smoothing（防止过度自信，提升泛化 +1-3%）
  - Sample Packing（短样本拼接，训练效率翻 2-3 倍）
  - 训练前自动数据清洗（去重+过滤 → 数据质量直接影响模型质量）
  - Cosine with restarts（多轮训练更平滑）
  - 训练后自动质量对比（测试题 A/B 对照）

Supported methods: SFT, DPO, ORPO, KTO
"""

from __future__ import annotations

import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from core import DATASETS_DIR, LORAS_DIR, log


# ═══ 早期 GPU 兼容性检查 ═══
# 必须在 transformers 的任何 attention 代码被执行之前运行
# RTX 5xxx (Blackwell, compute cap ≥ 10) 不兼容 xformers/flash_attn
def _early_gpu_compat_check():
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            if cap[0] >= 10:
                import os
                os.environ["XFORMERS_DISABLED"] = "1"
                # 提前标记 transformers 不要用 xformers
                try:
                    import transformers.utils
                    transformers.utils.is_xformers_available = lambda: False
                except Exception:
                    pass
                try:
                    import transformers.utils.import_utils as _iu
                    _iu.is_xformers_available = lambda: False
                except Exception:
                    pass
    except Exception:
        pass

_early_gpu_compat_check()


def _safe_update(task, p: float, msg: str):
    if task is not None:
        try:
            task.update_progress(float(p), str(msg))
        except Exception:
            pass


def _cleanup_vram(*objects):
    """训练后强制释放 GPU 显存。

    注意: Python 函数参数是引用的拷贝，del 局部变量不影响调用方。
    必须清空对象内部状态来释放 CUDA 张量。
    调用方仍应在调用后 del 自己的变量。
    """
    for obj in objects:
        try:
            # Trainer: 释放模型、优化器、数据
            if hasattr(obj, "model"):
                obj.model = None
            if hasattr(obj, "optimizer"):
                obj.optimizer = None
            if hasattr(obj, "lr_scheduler"):
                obj.lr_scheduler = None
            # Model: 移到 CPU 并释放参数
            if hasattr(obj, "cpu"):
                try:
                    obj.cpu()
                except Exception:
                    pass
            # Dataset: 释放 arrow 缓存
            if hasattr(obj, "cleanup_cache_files"):
                try:
                    obj.cleanup_cache_files()
                except Exception:
                    pass
        except Exception:
            pass
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if _check_npu_available():
            torch.npu.empty_cache()
    except Exception:
        pass


def _tok_kwarg(tok, trainer_cls=None):
    """TRL >= 0.14 / transformers >= 4.46 renamed tokenizer → processing_class.
    Safe check: inspect the actual class if provided, else check trl version.
    """
    if trainer_cls is not None:
        try:
            sig = inspect.signature(trainer_cls.__init__)
            if "processing_class" in sig.parameters:
                return {"processing_class": tok}
            if "tokenizer" in sig.parameters:
                return {"tokenizer": tok}
        except Exception:
            pass
    # Fallback: check trl version
    try:
        import trl
        ver = tuple(int(x) for x in trl.__version__.split(".")[:2])
        if ver >= (0, 14):
            return {"processing_class": tok}
    except Exception:
        pass
    # Last resort: check transformers version
    try:
        import transformers
        ver = tuple(int(x) for x in transformers.__version__.split(".")[:3])
        if ver >= (4, 46, 0):
            return {"processing_class": tok}
    except Exception:
        pass
    return {"tokenizer": tok}


def _filter_kwargs_for_callable(fn, kw: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return kw
        return {k: v for k, v in kw.items() if k in params}
    except Exception:
        return kw


def _first_keys(ds) -> set:
    try:
        ex = ds[0]
        return set(ex.keys()) if isinstance(ex, dict) else set()
    except Exception:
        return set()


def _require_deps(*mods: str):
    # 先确保 torchvision 不会炸掉 transformers 的 import 链
    try:
        from core import _TorchvisionGuardian
        tv = sys.modules.get("torchvision")
        if tv is not None:
            try:
                _ = tv.transforms.InterpolationMode
            except Exception:
                # 僵尸! 手动修复
                for f in sys.meta_path:
                    if isinstance(f, _TorchvisionGuardian):
                        f._ensure_available("torchvision")
                        f._ensure_available("torchvision.transforms")
                        break
    except Exception:
        pass

    missing = [m for m in mods if not _try_import(m)]
    if missing:
        raise RuntimeError(f"Missing: {', '.join(missing)}\npip install {' '.join(missing)}")


def _try_import(mod):
    try:
        __import__(mod)
        return True
    except ModuleNotFoundError:
        return False
    except ImportError as e:
        # 区分: 包本身缺失 vs 包内部依赖出错
        # "No module named 'xxx'" → 真正缺失
        if f"No module named '{mod}'" in str(e):
            return False
        # 其他 ImportError → 包存在但内部有问题，不应误报为缺失
        log(f"⚠️ {mod} 导入出错（已安装但有内部错误）: {e}")
        return True  # 包存在，让后续代码处理真正的错误
    except Exception as e:
        # AttributeError 等非 ImportError → 包存在但 import 链有 bug
        log(f"⚠️ {mod} 导入出错（非 ImportError）: {type(e).__name__}: {e}")
        return True  # 不误报为缺失


def _detect_gpu() -> Dict[str, Any]:
    info = {"cuda": False, "npu": False, "mps": False, "xpu": False,
            "bf16": False, "fp16": False, "flash_attn": False,
            "vram_mb": 0, "compute_cap": (0, 0),
            "device_count": 1, "device_type": "cpu", "name": "CPU"}
    try:
        import torch

        # ── NVIDIA CUDA ──
        if torch.cuda.is_available():
            info["cuda"] = True
            info["device_type"] = "cuda"
            info["device_count"] = torch.cuda.device_count()
            cap = torch.cuda.get_device_capability(0)
            info["compute_cap"] = cap
            info["bf16"] = cap[0] >= 8
            info["fp16"] = True
            info["vram_mb"] = torch.cuda.get_device_properties(0).total_mem // (1024 * 1024)
            info["name"] = torch.cuda.get_device_name(0)

            if info["device_count"] > 1:
                total_vram = sum(
                    torch.cuda.get_device_properties(i).total_mem // (1024 * 1024)
                    for i in range(info["device_count"])
                )
                names = [torch.cuda.get_device_name(i) for i in range(info["device_count"])]
                log(f"🎮 检测到 {info['device_count']} 块 GPU: {', '.join(names)} "
                    f"(总 VRAM: {total_vram // 1024}GB)")
                info["total_vram_mb"] = total_vram

            _gpu_too_new = cap[0] >= 10
            if _gpu_too_new:
                log(f"GPU compute capability {cap[0]}.{cap[1]} ≥ 10.0 "
                    f"(新架构)，flash_attn/xformers 暂不支持，将使用 SDPA")
                info["flash_attn"] = False
            else:
                try:
                    import flash_attn  # noqa
                    info["flash_attn"] = True
                except ImportError:
                    pass

        # ── 华为 Ascend NPU (torch_npu) ──
        elif _check_npu_available():
            import torch_npu  # noqa
            info["npu"] = True
            info["device_type"] = "npu"
            info["device_count"] = torch.npu.device_count()
            info["bf16"] = True
            info["fp16"] = True
            try:
                info["vram_mb"] = torch.npu.get_device_properties(0).total_memory // (1024 * 1024)
                info["name"] = torch.npu.get_device_name(0)
            except Exception:
                info["name"] = "Ascend NPU"
            log(f"🔷 检测到华为 Ascend NPU × {info['device_count']}")

        # ── Apple MPS ──
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["mps"] = True
            info["device_type"] = "mps"
            info["bf16"] = True
            info["fp16"] = True
            info["name"] = "Apple MPS"
            log("🍎 检测到 Apple MPS")

        # ── Intel XPU ──
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            info["xpu"] = True
            info["device_type"] = "xpu"
            info["device_count"] = torch.xpu.device_count()
            info["bf16"] = True
            info["fp16"] = True
            try:
                info["name"] = torch.xpu.get_device_name(0)
            except Exception:
                info["name"] = "Intel XPU"
            log(f"🔵 检测到 Intel XPU × {info['device_count']}")

    except Exception:
        pass
    return info


def _check_npu_available() -> bool:
    """检测华为 Ascend NPU 是否可用"""
    try:
        import torch
        import torch_npu  # noqa
        return torch.npu.is_available()
    except (ImportError, AttributeError):
        return False


def _disable_xformers_for_new_gpu(compute_cap: tuple):
    """彻底禁用 xformers + flash_attn 的所有 CUDA 操作。

    问题: RTX 5xxx (Blackwell, compute ≥ 10.0) 的 SM 架构太新,
    xformers 的 fa2/fa3/cutlass kernel 都不支持。
    但 xformers 可能已经被 import 并且函数引用已被缓存到各处。

    策略（按激进程度递增，全部执行）:
    1. 环境变量: 阻止未来的检测
    2. transformers 全局标记: 告知 transformers 不要用 xformers
    3. monkey-patch xformers.ops 核心函数: 让它们抛出 RuntimeError
       → transformers 的 attention 代码会 catch 并 fallback 到 sdpa/eager
    4. monkey-patch transformers 的 xformers 检测函数
    """
    import os
    import sys

    cap_str = f"{compute_cap[0]}.{compute_cap[1]}"
    _patched = []

    # ---- 1) 环境变量 ----
    os.environ["XFORMERS_DISABLED"] = "1"

    # ---- 2) transformers 全局标记 ----
    try:
        import transformers.utils
        # transformers 用 is_xformers_available() 判断是否启用
        # 直接 patch 返回 False
        transformers.utils.is_xformers_available = lambda: False
        _patched.append("is_xformers_available")
    except Exception:
        pass

    # 同时处理 import_utils 中的版本
    try:
        import transformers.utils.import_utils as _iu
        _iu.is_xformers_available = lambda: False
        _patched.append("import_utils.is_xformers_available")
    except Exception:
        pass

    # ---- 3) monkey-patch xformers 核心函数 ----
    # 这是最关键的: 即使代码已经 `from xformers.ops import xxx`
    # 这些函数对象本身被替换后，所有缓存的引用都会指向新函数

    def _xformers_disabled(*args, **kwargs):
        raise RuntimeError(
            f"xformers 已被 ForgeX 禁用 (GPU compute {cap_str} 不兼容)。"
            f"请使用 SDPA attention。"
        )

    # 核心: memory_efficient_attention 和 memory_efficient_attention_forward
    _xf_targets = [
        ("xformers.ops", "memory_efficient_attention"),
        ("xformers.ops", "memory_efficient_attention_forward"),
        ("xformers.ops.fmha", "memory_efficient_attention"),
        ("xformers.ops.fmha", "memory_efficient_attention_forward"),
        ("xformers.ops.fmha.dispatch", "memory_efficient_attention_forward"),
    ]
    for mod_name, func_name in _xf_targets:
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, func_name):
            setattr(mod, func_name, _xformers_disabled)
            _patched.append(f"{mod_name}.{func_name}")

    # ---- 4) patch transformers 的 xformers attention 路径 ----
    # transformers 各模型的 _attention_type == "xformers" 时会调用 xformers
    # patch 各模型的 XFormers attention class（如果已加载）
    _attn_mod_patterns = [
        "transformers.models.llama.modeling_llama",
        "transformers.models.qwen2.modeling_qwen2",
        "transformers.models.mistral.modeling_mistral",
        "transformers.models.gemma.modeling_gemma",
        "transformers.models.gemma2.modeling_gemma2",
        "transformers.models.phi3.modeling_phi3",
    ]
    for mod_name in _attn_mod_patterns:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        # 查找并 patch XFormers attention class
        for attr in dir(mod):
            if "xformers" in attr.lower() and "attention" in attr.lower():
                cls = getattr(mod, attr, None)
                if cls and isinstance(cls, type) and hasattr(cls, "forward"):
                    cls.forward = _xformers_disabled
                    _patched.append(f"{mod_name}.{attr}")

    # ---- 5) 防止 xformers 被重新正常 import ----
    # 用 import hook 拦截 xformers 的任何 import
    import importlib.abc
    import importlib.machinery

    class _XFormersBlocker(importlib.abc.MetaPathFinder):
        """阻止 xformers 被重新 import（返回空模块）"""
        _BLOCKED = {"xformers", "flash_attn"}

        def find_module(self, fullname, path=None):
            top = fullname.split(".")[0]
            if top in self._BLOCKED:
                return self
            return None

        def load_module(self, fullname):
            if fullname in sys.modules:
                return sys.modules[fullname]
            import types
            mod = types.ModuleType(fullname)
            mod.__path__ = []
            mod.__loader__ = self
            mod._disabled_by_forgex = True
            sys.modules[fullname] = mod
            return mod

    # 只在没有安装过 blocker 时安装
    if not any(isinstance(f, _XFormersBlocker) for f in sys.meta_path):
        sys.meta_path.insert(0, _XFormersBlocker())
        _patched.append("import_blocker")

    log(f"🛡️ GPU compute {cap_str}: xformers/flash_attn 已全面禁用 "
        f"({len(_patched)} patches: {', '.join(_patched[:5])}{'...' if len(_patched) > 5 else ''})")


_UNSLOTH_NUKED = False  # 全局标记：核弹是否已执行


def _nuke_unsloth_patches(force: bool = False):
    """核弹级清除: 从 Python 运行时彻底清除 Unsloth 的所有污染。

    ⚠️ 只在 Unsloth 模块确实存在于 sys.modules 时才执行破坏性操作!
    只清除被 Unsloth 实际污染的模块，不会误伤干净的 transformers 模块。

    Args:
        force: 强制执行（用于错误恢复场景，会重新加载模型）
    """
    global _UNSLOTH_NUKED
    if _UNSLOTH_NUKED and not force:
        _ensure_clean_trainer()
        return

    import sys
    import importlib

    _removed = []

    # ---- 0) 安全检查: Unsloth 是否真的存在? ----
    _unsloth_keys = [k for k in sys.modules if "unsloth" in k.lower()]
    if not _unsloth_keys and not force:
        # Unsloth 从未被加载过，不需要核弹清除
        log("Unsloth 清除: 跳过（sys.modules 中无 unsloth 模块）")
        _UNSLOTH_NUKED = True  # 标记为已处理，避免重复检查
        return

    # ---- 1) 删除所有 Unsloth 模块 ----
    for k in _unsloth_keys:
        del sys.modules[k]
    if _unsloth_keys:
        _removed.append(f"unsloth({len(_unsloth_keys)})")

    # ---- 2) 删除被 patch 的 transformers 模型模块 ----
    # 只删除确实被 Unsloth 污染的模块（检查 forward 方法是否被替换）
    _model_mod_keys = [k for k in sys.modules
                       if k.startswith("transformers.models.") and ".modeling_" in k]
    _actually_patched = []
    for k in _model_mod_keys:
        mod = sys.modules.get(k)
        if mod is None:
            continue
        # 检查模块中是否有被 Unsloth patch 的痕迹
        _is_patched = False
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name, None)
            if isinstance(attr, type):
                for method_name in ("forward", "_original_forward", "apply_qkv"):
                    method = getattr(attr, method_name, None)
                    if method is not None:
                        qn = getattr(method, "__qualname__", "") + getattr(method, "__module__", "")
                        if "unsloth" in qn.lower():
                            _is_patched = True
                            break
                if _is_patched:
                    break
        if _is_patched:
            _actually_patched.append(k)
            try:
                del sys.modules[k]
            except KeyError:
                pass
    if _actually_patched:
        _removed.append(f"patched_modeling({len(_actually_patched)})")

    # ---- 3) 清除 auto-mapping 模块缓存（只在有模块被删除时）----
    if _actually_patched:
        _auto_keys = [k for k in sys.modules
                      if "transformers" in k and ("auto_factory" in k or "modeling_auto" in k)]
        for k in _auto_keys:
            try:
                del sys.modules[k]
            except KeyError:
                pass
        if _auto_keys:
            _removed.append(f"auto({len(_auto_keys)})")

        # ---- 4) 清除 LazyAutoMapping 缓存 ----
        try:
            from transformers.models.auto import modeling_auto as _ma
            for _mapping_name in dir(_ma):
                _mapping = getattr(_ma, _mapping_name, None)
                if hasattr(_mapping, "_modules"):
                    try:
                        _mapping._modules = {}
                    except Exception:
                        pass
        except Exception:
            pass

    # ---- 5) 重载 Trainer ----
    _ensure_clean_trainer()
    _removed.append("trainer")

    _UNSLOTH_NUKED = True

    if _removed:
        log(f"🧹 Unsloth 清除完成: {', '.join(_removed)}")
    else:
        log("Unsloth 清除: 无需操作")


def _ensure_clean_trainer():
    """轻量级: 只检查并修复 Trainer 的 Unsloth patch。

    可以在模型加载后安全调用，不会破坏已加载的模块。
    """
    import sys
    import importlib

    _trainer_key = "transformers.trainer"
    if _trainer_key not in sys.modules:
        return

    try:
        from transformers import Trainer as _HFTrainer
        _needs_fix = False
        for attr in ("_inner_training_loop", "compute_loss", "training_step"):
            fn = getattr(_HFTrainer, attr, None)
            if fn is not None:
                qn = getattr(fn, "__qualname__", "") + getattr(fn, "__name__", "")
                if "unsloth" in qn.lower() or "fast_inner" in qn.lower():
                    _needs_fix = True
                    break

        if _needs_fix:
            importlib.reload(sys.modules[_trainer_key])
            log("✅ Trainer Unsloth patch 已清除（重载）")
    except Exception as e:
        log(f"⚠️ 清理 Trainer patch 时出错: {e}")


@dataclass
class MethodSupport:
    method: str
    available: bool
    reason: str = ""


class TrainerEngine:

    def available_methods(self) -> List[MethodSupport]:
        out = [MethodSupport("sft", True, "")]
        for cls_name, name in [("DPOTrainer", "dpo"), ("ORPOTrainer", "orpo"), ("KTOTrainer", "kto")]:
            try:
                __import__("trl", fromlist=[cls_name])
                out.append(MethodSupport(name, True, ""))
            except Exception as e:
                out.append(MethodSupport(name, False, str(e)))
        return out

    def train(self, method: str, backend: str, base_model: str,
              dataset_path: Any, params: Dict[str, Any], task=None):
        method = (method or "sft").lower().strip()
        support = {m.method: m for m in self.available_methods()}
        if method not in support:
            raise RuntimeError(f"Unknown method: {method}")
        if not support[method].available:
            raise RuntimeError(f"'{method}' disabled: {support[method].reason}")

        # 发送训练开始事件
        try:
            from core.event_bus import EventBus, Events
            EventBus.emit(Events.TRAIN_START, method=method, base_model=base_model)
        except Exception:
            pass

        dispatch = {
            "sft": self._train_sft,
            "dpo": lambda *a, **kw: self._train_dpo_like("dpo", *a, **kw),
            "orpo": lambda *a, **kw: self._train_dpo_like("orpo", *a, **kw),
            "kto": self._train_kto,
        }
        try:
            result = dispatch[method](base_model, dataset_path, params, task=task, backend=backend or "trl")
            try:
                EventBus.emit(Events.TRAIN_COMPLETE, method=method, result=str(result))
            except Exception:
                pass
            return result
        except Exception as e:
            try:
                EventBus.emit(Events.TRAIN_ERROR, method=method, error=str(e))
            except Exception:
                pass
            raise

    # ================================================================
    #  共用基础设施
    # ================================================================

    def _resolve_dataset_paths(self, dataset_path: Any) -> List[Path]:
        items = dataset_path
        if isinstance(items, (str, Path)):
            items = [items]
        out: List[Path] = []
        for item in (items or []):
            raw = str(item)
            p = Path(raw).expanduser()
            if not p.is_absolute():
                cand = Path(DATASETS_DIR) / raw
                if cand.exists():
                    p = cand
                else:
                    p = (Path(__file__).resolve().parent.parent / raw)
            if not p.exists():
                raise FileNotFoundError(f"Dataset not found: {raw} -> {p}")
            out.append(p)
        return out

    def _load_dataset_robust(self, ds_paths: List[Path], task=None):
        """健壮的数据集加载：自动处理 JSON/JSONL/CSV/Parquet/TXT 各种格式。

        加载策略（按格式分）：
        - .jsonl / .json → load_dataset("json") → 手动 JSON 解析 fallback
        - .csv            → load_dataset("csv") → pandas fallback
        - .parquet        → load_dataset("parquet")
        - .txt            → 逐行读取为 {"text": line}
        多文件时按格式分组加载后合并。
        """
        files = [str(x) for x in ds_paths]
        _safe_update(task, 21, f"加载 {len(files)} 个文件: {[Path(f).name for f in files]}")

        from datasets import Dataset, concatenate_datasets

        all_datasets = []

        for fpath in files:
            p = Path(fpath)
            ext = p.suffix.lower()
            ds = None

            # ---- 按格式分派 ----
            if ext in (".parquet",):
                try:
                    from datasets import load_dataset
                    ds = load_dataset("parquet", data_files=str(p), split="train")
                except Exception as e:
                    log(f"Parquet 加载失败 {p.name}: {e}")

            elif ext in (".csv", ".tsv"):
                # 方案 A: datasets
                try:
                    from datasets import load_dataset
                    sep = "\t" if ext == ".tsv" else ","
                    ds = load_dataset("csv", data_files=str(p), split="train",
                                      delimiter=sep)
                except Exception:
                    pass
                # 方案 B: pandas fallback
                if ds is None:
                    try:
                        import pandas as pd
                        sep = "\t" if ext == ".tsv" else ","
                        df = pd.read_csv(p, sep=sep, encoding="utf-8-sig")
                        df = df.fillna("")
                        ds = Dataset.from_pandas(df)
                    except Exception as e:
                        log(f"CSV 加载失败 {p.name}: {e}")

            elif ext in (".txt",):
                try:
                    raw = p.read_text(encoding="utf-8-sig").strip()
                    lines = [l.strip() for l in raw.splitlines() if l.strip()]
                    if lines:
                        ds = Dataset.from_dict({"text": lines})
                except Exception as e:
                    log(f"TXT 加载失败 {p.name}: {e}")

            else:
                # JSON / JSONL（默认）
                ds = self._load_json_robust(p, task)

            if ds is not None and len(ds) > 0:
                all_datasets.append(ds)
                _safe_update(task, 22, f"✅ {p.name}: {len(ds)} 条")
            else:
                log(f"⚠️ 文件 {p.name} 加载后为空或失败")

        if not all_datasets:
            raise RuntimeError(
                f"数据集全部加载失败。文件: {[Path(f).name for f in files]}\n"
                "支持的格式: .jsonl, .json, .csv, .tsv, .parquet, .txt"
            )

        # 合并多文件（统一列名）
        if len(all_datasets) == 1:
            return all_datasets[0]

        # 多文件合并：取所有数据集的列名交集
        common_cols = set(all_datasets[0].column_names)
        for d in all_datasets[1:]:
            common_cols &= set(d.column_names)
        if not common_cols:
            # 无交集，尝试都转为 text 列
            merged_texts = []
            for d in all_datasets:
                if "text" in d.column_names:
                    merged_texts.extend(d["text"])
                else:
                    # 把第一个字段当 text
                    first_col = d.column_names[0]
                    merged_texts.extend([str(x) for x in d[first_col]])
            return Dataset.from_dict({"text": merged_texts})

        # 保留公共列后合并
        aligned = []
        for d in all_datasets:
            d = d.remove_columns([c for c in d.column_names if c not in common_cols])
            aligned.append(d)
        try:
            merged = concatenate_datasets(aligned)
        except Exception as e_concat:
            # 类型冲突（如 int64 vs float64）→ 统一列类型后重试
            log(f"concatenate_datasets 类型冲突: {e_concat}，尝试统一类型...")
            from datasets import Features, Value, Sequence
            # 取第一个数据集的 features 作为基准，把数字类型统一为 float64
            _base_features = aligned[0].features.copy()
            for fname, ftype in _base_features.items():
                if hasattr(ftype, 'dtype') and 'int' in str(ftype):
                    _base_features[fname] = Value('float64')
            try:
                _casted = [d.cast(_base_features) for d in aligned]
                merged = concatenate_datasets(_casted)
            except Exception:
                # 终极 fallback: 全部转文本合并
                log(f"类型统一也失败，降级为纯文本合并")
                merged_texts = []
                for d in aligned:
                    for i in range(len(d)):
                        row = {k: d[i][k] for k in d.column_names}
                        merged_texts.append(json.dumps(row, ensure_ascii=False, default=str))
                merged = Dataset.from_dict({"text": merged_texts})
        _safe_update(task, 22, f"合并完成: {len(merged)} 条 (来自 {len(files)} 个文件)")
        return merged

    def _load_json_robust(self, p: Path, task=None):
        """JSON/JSONL 文件健壮加载"""
        from datasets import Dataset

        # 方案 A：标准 load_dataset
        try:
            from datasets import load_dataset
            ds = load_dataset("json", data_files=str(p), split="train")
            if len(ds) > 0:
                return ds
        except Exception as e1:
            log(f"load_dataset(json) failed for {p.name}: {e1}")

        # 方案 B：手动解析
        try:
            raw = p.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            raw = p.read_text(encoding="utf-8", errors="replace")

        raw = raw.strip()
        if not raw:
            return None

        all_rows = []

        # JSON array
        if raw.startswith("["):
            try:
                rows = json.loads(raw)
                if isinstance(rows, list):
                    all_rows.extend([r for r in rows if isinstance(r, dict)])
            except json.JSONDecodeError:
                pass

        # JSONL / 逐行
        if not all_rows:
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        all_rows.append(obj)
                    elif isinstance(obj, list):
                        all_rows.extend([r for r in obj if isinstance(r, dict)])
                except json.JSONDecodeError:
                    continue

        if not all_rows:
            return None

        # 统一字段 + 处理类型冲突
        all_keys = set()
        for row in all_rows:
            all_keys.update(row.keys())

        columns = {k: [] for k in all_keys}
        for row in all_rows:
            for k in all_keys:
                val = row.get(k)
                if k in ("messages", "conversations") and isinstance(val, list):
                    columns[k].append(val)
                elif isinstance(val, (list, dict)):
                    columns[k].append(json.dumps(val, ensure_ascii=False, default=str))
                else:
                    columns[k].append(val if val is not None else "")

        # ═══ 修复 datasets 类型推断冲突 ═══
        # 常见问题：
        #   1. 同列混合 int/float (如 score: 5 和 score: 4.5) → datasets 报类型不一致
        #   2. 空列表 [] 推断为 List(null)，非空列表推断为 List(string) → 冲突
        #   3. 某些行有值，某些行是 "" → 混合 string 和 numeric
        for k, vals in columns.items():
            if not vals:
                continue
            # 检测列中是否有混合 int/float
            _has_int = any(isinstance(v, int) and not isinstance(v, bool) for v in vals)
            _has_float = any(isinstance(v, float) for v in vals)
            if _has_int and _has_float:
                # 全部统一为 float
                columns[k] = [float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v for v in vals]
            elif _has_int or _has_float:
                # 检查是否有空字符串混入数字列（None 被替换为 "" 的情况）
                _has_empty_str = any(v == "" for v in vals)
                if _has_empty_str:
                    columns[k] = [
                        float(v) if isinstance(v, (int, float)) and not isinstance(v, bool)
                        else None  # 空字符串 → None，让 datasets 处理为 null
                        for v in vals
                    ]

        try:
            return Dataset.from_dict(columns)
        except Exception as e_dict:
            # 如果仍然失败，把所有非核心列转为字符串
            log(f"Dataset.from_dict 类型冲突: {e_dict}，尝试转为字符串...")
            _core_keys = {"messages", "conversations", "instruction", "input", "output",
                         "text", "prompt", "response", "question", "answer",
                         "chosen", "rejected", "completion", "label"}
            for k, vals in columns.items():
                if k not in _core_keys:
                    columns[k] = [json.dumps(v, ensure_ascii=False, default=str)
                                  if not isinstance(v, str) else v for v in vals]
            return Dataset.from_dict(columns)

    def _common_args(self, out_dir: Path, p: Dict[str, Any]) -> Dict[str, Any]:
        """构建通用 training kwargs dict.
        
        显存优化策略：
        1. paged_adamw_8bit: 优化器状态省 ~75% 显存（对比 AdamW FP32）
        2. gradient_checkpointing: 省 ~60% 激活内存（已在 _load_model_tok 启用）
        3. gradient_accumulation: 小 batch + 累积 = 等效大 batch，不多占显存
        4. dataloader_num_workers=0: 避免 Windows 多进程 bug + 省 RAM
        5. optim_target_modules: 只对 LoRA 层保留完整优化器状态（TRL >= 0.12）
        """
        gpu = _detect_gpu()

        # Optimizer — 有加速器就用 paged_adamw_8bit（节省 ~75% 优化器显存）
        optim = (p.get("optim") or "").strip()
        _has_accelerator = gpu.get("cuda") or gpu.get("npu") or gpu.get("xpu")
        if not optim:
            if _has_accelerator:
                optim = "paged_adamw_8bit"
            else:
                optim = "adamw_torch"
        if ("8bit" in optim.lower() or "paged" in optim.lower()) and not _has_accelerator:
            optim = "adamw_torch"

        # Precision
        _fp16, _bf16 = p.get("fp16"), p.get("bf16")
        if _fp16 is None and _bf16 is None:
            if gpu["bf16"]:     _bf16, _fp16 = True, False
            elif _has_accelerator: _fp16, _bf16 = True, False
            else:               _fp16, _bf16 = False, False
        else:
            _fp16, _bf16 = bool(_fp16 or False), bool(_bf16 or False)

        report_to = p.get("report_to", "none")
        if isinstance(report_to, str):
            report_to = [report_to] if report_to != "none" else []

        ms = p.get("max_steps", None)
        args = dict(
            output_dir=str(out_dir),
            per_device_train_batch_size=int(p.get("batch_size", 1)),
            gradient_accumulation_steps=int(p.get("gradient_accumulation_steps", 4)),
            learning_rate=float(p.get("lr", 2e-4)),
            num_train_epochs=float(p.get("epochs", 1)),
            max_steps=int(ms) if ms not in (None, "", 0) else -1,
            lr_scheduler_type=p.get("lr_scheduler_type", "cosine"),
            warmup_ratio=float(p.get("warmup_ratio", 0.03)),
            weight_decay=float(p.get("weight_decay", 0.01)),
            max_grad_norm=float(p.get("max_grad_norm", 1.0)),
            label_smoothing_factor=float(p.get("label_smoothing", 0.1)),
            logging_steps=int(p.get("logging_steps", 5)),
            save_steps=int(p.get("save_steps", 500)),
            save_total_limit=int(p.get("save_total_limit", 2)),  # 2 而非 3，省磁盘
            fp16=_fp16, bf16=_bf16,
            report_to=report_to,
            remove_unused_columns=False,
            optim=optim,
            dataloader_pin_memory=bool(_has_accelerator),   # 有加速器时才 pin
            dataloader_num_workers=0,             # Windows 兼容 + 省 RAM
        )

        # optim_target_modules — 只对 LoRA 参数保留优化器状态（减少显存）
        # 注意：这个参数只有部分 transformers/trl 版本支持
        try:
            from transformers import TrainingArguments as _TA
            import inspect
            _ta_params = inspect.signature(_TA.__init__).parameters
            if "optim_target_modules" in _ta_params:
                args["optim_target_modules"] = [r".*lora.*"]
        except Exception:
            pass

        return args

    def _load_model_tok(self, base_model: str, p: Dict[str, Any]):
        _require_deps("transformers")

        # ====== LoRA 目录自动检测 ======
        # 如果 base_model 指向 LoRA 适配器目录（有 adapter_config.json 但没有 config.json）
        # 自动读取真正的基座模型路径
        bm_path = Path(base_model) if not base_model.startswith(("http://", "https://")) else None
        if bm_path and bm_path.is_dir():
            acfg_file = bm_path / "adapter_config.json"
            has_config = (bm_path / "config.json").exists()
            has_weights = bool(list(bm_path.glob("*.safetensors")) + list(bm_path.glob("model*.bin")))
            # 纯 LoRA: 有 adapter 没 config；或有 adapter 没权重
            if acfg_file.exists() and not (has_config and has_weights):
                try:
                    acfg = json.loads(acfg_file.read_text(encoding="utf-8"))
                    real_base = acfg.get("base_model_name_or_path", "")
                    if real_base:
                        log(f"⚠️ 检测到 LoRA 适配器目录: {base_model}")
                        # 修复过时路径
                        try:
                            from core.distiller import _repair_base_model_path
                            resolved = _repair_base_model_path(real_base)
                            if resolved:
                                if resolved != real_base:
                                    log(f"   🔧 路径修复: {real_base} → {resolved}")
                                base_model = resolved
                            else:
                                base_model = real_base  # 让后续加载尝试
                        except ImportError:
                            base_model = real_base
                        log(f"   自动切换到基座模型: {base_model}")
                    else:
                        raise ValueError(
                            f"LoRA 适配器 {bm_path.name} 的 adapter_config.json 中缺少 base_model_name_or_path。\n"
                            f"请直接填写基座模型路径（如 Qwen/Qwen2.5-7B-Instruct）"
                        )
                except json.JSONDecodeError as e:
                    raise ValueError(f"adapter_config.json 解析失败: {e}")

        gpu = _detect_gpu()

        # ═══ 新架构 GPU 兼容性保护 ═══
        # RTX 5xxx (Blackwell, compute cap ≥ 10.0):
        # xformers/flash_attn 的 CUDA kernel 不支持 SM ≥ 100
        #
        # 为什么删 sys.modules 不够: 代码已经通过 `from xformers.ops import xxx`
        # 缓存了函数引用。必须直接 monkey-patch 那些会被调用的函数。
        if gpu["cuda"] and gpu["compute_cap"][0] >= 10:
            _disable_xformers_for_new_gpu(gpu["compute_cap"])

        # ====== 优先尝试 Unsloth（2-5x 加速 + 70% 省内存）======
        qlora = bool(p.get("use_qlora") or p.get("q_lora"))
        use_unsloth = bool(p.get("use_unsloth", True))  # 默认尝试
        max_seq = int(p.get("max_seq_len", p.get("max_seq_length", 2048)))
        _unsloth_loaded = False

        if use_unsloth and gpu["cuda"]:
            # 新架构 GPU 跳过 Unsloth（依赖 xformers/flash_attn 的 kernel 不兼容）
            if gpu["compute_cap"][0] >= 10:
                _safe_update(p.get("_task"), 3,
                    f"⚡ GPU compute {gpu['compute_cap'][0]}.{gpu['compute_cap'][1]} "
                    f"暂不兼容 Unsloth（xformers/triton），使用标准 HuggingFace")
                use_unsloth = False

        if use_unsloth and gpu["cuda"]:
            try:
                from unsloth import FastLanguageModel
                _safe_update(p.get("_task"), 3, "🦥 检测到 Unsloth，优先使用（2x 加速 + 70% 省显存）")
                model, tok = FastLanguageModel.from_pretrained(
                    model_name=base_model,
                    max_seq_length=max_seq,
                    dtype=None,  # auto
                    load_in_4bit=qlora,
                    trust_remote_code=True,
                )
                rank = int(p.get("rank", 64))
                alpha = int(p.get("alpha", rank * 2))
                # "all-linear" 在某些 Unsloth 版本中会被 set() 拆成单字符
                # 用显式列表替代，覆盖主流模型的所有线性层
                _unsloth_targets = [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ]
                model = FastLanguageModel.get_peft_model(
                    model, r=rank, lora_alpha=alpha,
                    target_modules=_unsloth_targets,
                    lora_dropout=float(p.get("lora_dropout", 0.05)),
                    bias="none",
                    use_gradient_checkpointing="unsloth",
                    max_seq_length=max_seq,
                )
                if tok.pad_token is None: tok.pad_token = tok.eos_token
                if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
                tok.padding_side = "right"
                trainable = sum(pp.numel() for pp in model.parameters() if pp.requires_grad)
                total = sum(pp.numel() for pp in model.parameters())
                _safe_update(p.get("_task"), 8,
                    f"🦥 Unsloth | LoRA r={rank} α={alpha} | "
                    f"trainable {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")
                p["_backend_unsloth"] = True
                _unsloth_loaded = True
                return model, tok
            except (ImportError, ModuleNotFoundError) as e:
                log(f"Unsloth 不可用（{e}），降级为 HuggingFace")
            except Exception as e:
                _safe_update(p.get("_task"), 3, f"Unsloth 加载失败，降级为 HuggingFace: {e}")
                log(f"Unsloth fallback: {e}")
                # 释放 Unsloth 创建的半成品模型（可能已消耗大量显存）
                try:
                    del model, tok
                except NameError:
                    pass
                try:
                    import torch as _t
                    import gc as _gc
                    _gc.collect()
                    if _t.cuda.is_available():
                        _t.cuda.empty_cache()
                except Exception:
                    pass

        # ====== Unsloth 清理（仅在 Unsloth 被加载过时执行）======
        # _nuke_unsloth_patches 内部已有安全检查:
        # - 如果 sys.modules 中没有 unsloth 模块，会自动跳过
        # - 只清除被实际污染的模块，不会破坏干净的 auto-mapping
        if not _unsloth_loaded:
            _nuke_unsloth_patches()

        # ====== 标准 HuggingFace 路径 ======
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Tokenizer
        tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        tok.padding_side = "right"

        # Model kwargs
        mk: Dict[str, Any] = dict(trust_remote_code=True)
        _device_map = None
        _has_accelerator = gpu.get("cuda") or gpu.get("npu") or gpu.get("mps") or gpu.get("xpu")

        if _has_accelerator:
            if gpu.get("cuda"):
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        _ = torch.zeros(1, device="cuda")
                        del _
                        if gpu.get("device_count", 1) > 1:
                            # 多卡: 用 device_map="auto" 让 accelerate 自动分配
                            _device_map = "auto"
                            _safe_update(p.get("_task"), 3,
                                f"🎮 多 GPU 模式: {gpu['device_count']} 块 GPU, device_map='auto'")
                        else:
                            _device_map = {"": 0}
                except Exception:
                    _device_map = "auto"
            elif gpu.get("npu"):
                try:
                    import torch
                    import torch_npu  # noqa
                    torch.npu.empty_cache()
                    if gpu.get("device_count", 1) > 1:
                        _device_map = "auto"
                        _safe_update(p.get("_task"), 3,
                            f"🔷 多 NPU 模式: {gpu['device_count']} 块 NPU")
                    else:
                        _device_map = {"": 0}
                except Exception:
                    _device_map = "auto"
            elif gpu.get("mps"):
                _device_map = {"": "mps"}
            elif gpu.get("xpu"):
                if gpu.get("device_count", 1) > 1:
                    _device_map = "auto"
                else:
                    _device_map = {"": 0}

            mk["device_map"] = _device_map
        if gpu["flash_attn"]:
            mk["attn_implementation"] = "flash_attention_2"
            _safe_update(p.get("_task"), 3, "✅ FlashAttention 2 已启用")
        elif gpu["compute_cap"][0] >= 10:
            # Blackwell 等新架构: flash_attn/xformers 都不支持
            # 强制 sdpa（PyTorch 原生），避免 transformers 自动选到 xformers
            mk["attn_implementation"] = "sdpa"
            _safe_update(p.get("_task"), 3,
                f"⚡ 使用 SDPA attention（GPU compute {gpu['compute_cap'][0]}.{gpu['compute_cap'][1]} "
                f"暂不兼容 flash_attn/xformers）")
        try:
            import torch
            from core.safe_loader import dtype_kwarg
            _dtype = torch.bfloat16 if gpu["bf16"] else (torch.float16 if _has_accelerator else torch.float32)
            mk.update(dtype_kwarg(_dtype))
        except Exception:
            pass

        # QLoRA
        qlora = bool(p.get("use_qlora") or p.get("q_lora"))
        if qlora:
            try:
                from transformers import BitsAndBytesConfig
                import torch
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if gpu["bf16"] else torch.float16,
                )
                mk["quantization_config"] = bnb
            except Exception as e:
                raise RuntimeError(f"QLoRA requires bitsandbytes: {e}")

        # Load (with multi-level fallback)
        # 使用 safe_load_model 代替直接调用 AutoModelForCausalLM.from_pretrained
        # 解决 "Could not find LlamaForCausalLM" 等 LazyAutoMapping 失效问题
        from core.safe_loader import safe_load_model as _safe_load

        def _try_load(extra_msg=""):
            return _safe_load(base_model, **mk)

        try:
            model = _try_load()
        except Exception as e:
            msg = str(e).lower()
            _handled = False

            # Fallback A: 模型不支持当前 attn_implementation → 降级
            # Phi-3 等模型不支持 sdpa，报 "does not support an attention implementation
            # through torch.nn.functional.scaled_dot_product_attention"
            _attn_impl = mk.get("attn_implementation")
            if _attn_impl and "does not support" in str(e) and "attention" in msg:
                _fallback_attn = "eager" if _attn_impl != "eager" else None
                if _fallback_attn:
                    _safe_update(p.get("_task"), 5,
                        f"⚠️ 模型不支持 {_attn_impl} attention，降级为 {_fallback_attn}")
                    log(f"attn_implementation fallback: {_attn_impl} → {_fallback_attn}")
                    mk["attn_implementation"] = _fallback_attn
                    try:
                        model = _try_load()
                        _handled = True
                    except Exception as e_attn:
                        # eager 也失败 → 去掉 attn_implementation 让 transformers 自己选
                        log(f"attn_implementation={_fallback_attn} 也失败: {e_attn}")
                        mk.pop("attn_implementation", None)
                        try:
                            model = _try_load()
                            _handled = True
                        except Exception:
                            pass  # 继续后续 fallback

            # Fallback 0: trust_remote_code 模型缺少自定义 .py 文件
            if not _handled and "does not appear to have a file named" in str(e) and ".py" in str(e):
                _safe_update(p.get("_task"), 5, "⚠️ 检测到缺少自定义模型代码，尝试自动修复...")
                try:
                    from core.merger import _copy_custom_model_code
                    bm_path = Path(base_model)
                    if bm_path.is_dir():
                        _cfg = json.loads((bm_path / "config.json").read_text(encoding="utf-8"))
                        src = _cfg.get("_name_or_path", "")
                        recipe_f = bm_path / "forgex_merge_recipe.json"
                        if not src and recipe_f.exists():
                            src = json.loads(recipe_f.read_text(encoding="utf-8")).get("base_model", "")
                        if src:
                            _copy_custom_model_code(src, bm_path)
                            model = _try_load()
                            _safe_update(p.get("_task"), 8, "✅ 自定义代码修复成功")
                            _handled = True
                        else:
                            raise
                    else:
                        raise
                except Exception as e_code:
                    if "does not appear to have a file named" in str(e_code):
                        raise RuntimeError(
                            f"模型目录缺少自定义代码文件（如 modeling_phi3.py）。\n"
                            f"这通常发生在合并模型时未复制源模型的 .py 文件。\n"
                            f"修复方法: 从原始模型目录（如 HuggingFace 缓存）手动复制对应的 .py 文件到:\n"
                            f"  {base_model}\n"
                            f"原始错误: {e}"
                        ) from e
                    if not _handled:
                        raise

            # 检测 triton/bitsandbytes 相关错误（Windows 上 triton 不可用）
            if not _handled:
                _triton_or_bnb = any(k in msg for k in [
                    "frozenset", "bitsandbytes", "bnb", "4bit", "quantiz",
                    "libcudart", "triton", "no kernel", "cextension",
                    "no module named 'triton", "no module named 'bitsandbytes"])

                # Fallback 1: QLoRA / triton 失败 → 降级普通 LoRA
                bnb_fail = qlora and _triton_or_bnb
                if not bnb_fail and _triton_or_bnb:
                    bnb_fail = True
                if bnb_fail:
                    _safe_update(p.get("_task"), 5, "⚠️ QLoRA/triton 不可用，降级为普通 LoRA")
                    mk.pop("quantization_config", None)
                    p["use_qlora"] = False
                    try:
                        model = _try_load()
                        _handled = True
                    except Exception as e_bnb:
                        log(f"BNB fallback also failed: {e_bnb}")
                        if not any(k in msg for k in ["cuda error", "device-side assert",
                                   "out of memory", "oom"]):
                            raise RuntimeError(
                                f"模型加载失败。原始错误: {e}\nQLoRA 降级也失败: {e_bnb}"
                            ) from e_bnb

                # Fallback 2: CUDA assert / device_map 问题 → 尝试 device_map="auto"
                cuda_fail = any(k in msg for k in [
                    "cuda error", "device-side assert", "cudamemgetinfo",
                    "caching_allocator", "out of memory", "oom",
                    "accelerator", "device_map"])
                if not _handled and cuda_fail and mk.get("device_map") == {"": 0}:
                    _safe_update(p.get("_task"), 5, "⚠️ CUDA 加载失败，切换 device_map='auto'...")
                    import torch
                    torch.cuda.empty_cache()
                    mk["device_map"] = "auto"
                    try:
                        model = _try_load()
                        _handled = True
                    except Exception as e2:
                        _safe_update(p.get("_task"), 5, "⚠️ device_map='auto' 也失败，尝试 CPU 加载...")
                        mk.pop("device_map", None)
                        mk.pop("quantization_config", None)
                        p["use_qlora"] = False
                        try:
                            model = _try_load()
                            if torch.cuda.is_available():
                                model = model.to("cuda:0")
                            _handled = True
                        except Exception as e3:
                            raise RuntimeError(
                                f"模型加载失败（已尝试 3 种方式）。\n"
                                f"原始错误: {e}\n"
                                f"device_map='auto': {e2}\n"
                                f"CPU fallback: {e3}\n"
                                f"建议：1) 重启程序清理 CUDA 状态  2) 检查 PyTorch/CUDA 驱动版本匹配"
                            ) from e3

                if not _handled:
                    raise

        # Gradient checkpointing
        if bool(p.get("gradient_checkpointing", True)) and hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                model.gradient_checkpointing_enable()  # transformers < 4.36
            model.config.use_cache = False

        # ═══ 安全验证: 检查模型是否被 Unsloth 污染 ═══
        # _nuke_unsloth_patches 在 from_pretrained 之前已删除所有被污染的模块，
        # 此处只做验证（不应触发）
        if not p.get("_backend_unsloth"):
            _contaminated = False
            for _name, _module in model.named_modules():
                if hasattr(_module, "apply_qkv") or hasattr(type(_module), "apply_qkv"):
                    _contaminated = True
                    break
            if _contaminated:
                # 极端情况: nuke 后仍然被污染 → 再次清理并重新加载
                log("⚠️ 模型加载后仍检测到 Unsloth 污染，执行二次清理...")
                _nuke_unsloth_patches(force=True)
                del model
                import gc; gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                # 重新加载（使用 safe_load 防止映射失效）
                model = _safe_load(base_model, **mk)
                _safe_update(p.get("_task"), 7, "🔧 二次清理完成，模型已重新加载")

            # 设置 max_seq_length 安全属性（某些代码路径可能引用它）
            if not hasattr(model, "max_seq_length"):
                model.max_seq_length = max_seq

        # LoRA (with DoRA + rsLoRA support)
        _require_deps("peft")
        from peft import LoraConfig, get_peft_model

        targets = p.get("target_modules")
        if not targets:
            try:
                from core.utils import get_lora_targets
                targets = get_lora_targets(base_model)
            except Exception:
                targets = "all-linear"  # peft: 自动匹配所有 nn.Linear

        rank = int(p.get("rank", 64))
        alpha = int(p.get("alpha", rank * 2))

        # DoRA: Weight-Decomposed LoRA — 论文证实同 rank 下一致优于标准 LoRA
        # 原理: 将权重分解为 magnitude + direction, LoRA 只调 direction
        use_dora = bool(p.get("use_dora", True))  # 默认启用

        # rsLoRA: Rank-Stabilized scaling — rank 越高越稳定
        # 原理: lora_alpha 自动除以 sqrt(r) 而非 r
        use_rslora = bool(p.get("use_rslora", True))  # 默认启用

        # 检测 peft 版本是否支持
        _dora_ok, _rslora_ok = False, False
        try:
            import inspect as _ins
            _lora_params = set(_ins.signature(LoraConfig.__init__).parameters.keys())
            _dora_ok = "use_dora" in _lora_params
            _rslora_ok = "use_rslora" in _lora_params
        except Exception:
            pass

        _extra_lora_kw = {}
        if use_dora and _dora_ok:
            _extra_lora_kw["use_dora"] = True
        if use_rslora and _rslora_ok:
            _extra_lora_kw["use_rslora"] = True

        _dora_status = "DoRA✅" if _extra_lora_kw.get("use_dora") else "DoRA❌(peft版本不支持)" if use_dora else "DoRA关"
        _rslora_status = "rsLoRA✅" if _extra_lora_kw.get("use_rslora") else "rsLoRA❌(peft版本不支持)" if use_rslora else "rsLoRA关"
        _safe_update(p.get("_task"), 7, f"LoRA 配置: {_dora_status} | {_rslora_status}")

        # ═══ MoLoRA (多专家 LoRA) 模式 ═══
        _use_molora = bool(p.get("use_molora", False))
        if _use_molora:
            from core.molora import MoLoRAConfig, apply_molora
            _n_experts = int(p.get("molora_n_experts", 4))
            _molora_topk = int(p.get("molora_top_k", 2))
            _expert_labels = p.get("molora_expert_labels", [f"专家{i}" for i in range(_n_experts)])

            molora_cfg = MoLoRAConfig(
                n_experts=_n_experts,
                rank=rank,
                alpha=float(alpha),
                top_k=_molora_topk,
                dropout=float(p.get("lora_dropout", 0.05)),
                target_modules=targets if isinstance(targets, list) else [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
                aux_loss_weight=float(p.get("molora_aux_weight", 0.01)),
                expert_labels=_expert_labels,
            )
            model = apply_molora(model, molora_cfg)
            _safe_update(p.get("_task"), 8,
                f"🧠 MoLoRA {_n_experts}专家 × r{rank} Top-{_molora_topk} | "
                f"专家: {', '.join(_expert_labels[:4])}")
            return model, tok

        # ═══ 标准 LoRA (PEFT) ═══
        try:
            lora_cfg = LoraConfig(
                r=rank, lora_alpha=alpha,
                target_modules=targets,
                lora_dropout=float(p.get("lora_dropout", 0.05)),
                bias="none", task_type="CAUSAL_LM",
                **_extra_lora_kw,
            )
            model = get_peft_model(model, lora_cfg)
        except (ValueError, KeyError, TypeError) as e:
            # 降级: 先去掉 DoRA/rsLoRA，再降级 targets
            log(f"LoRA 高级特性不兼容，降级: {e}")
            try:
                lora_cfg = LoraConfig(
                    r=rank, lora_alpha=alpha,
                    target_modules=targets,
                    lora_dropout=float(p.get("lora_dropout", 0.05)),
                    bias="none", task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, lora_cfg)
            except (ValueError, KeyError):
                targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                lora_cfg = LoraConfig(
                    r=rank, lora_alpha=alpha, target_modules=targets,
                    lora_dropout=float(p.get("lora_dropout", 0.05)),
                    bias="none", task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, lora_cfg)

        trainable = sum(pp.numel() for pp in model.parameters() if pp.requires_grad)
        total = sum(pp.numel() for pp in model.parameters())
        _safe_update(p.get("_task"), 8,
            f"LoRA r={rank} α={alpha} {_dora_status} {_rslora_status} | "
            f"trainable {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")
        return model, tok

    def _callbacks(self, task, has_eval=False):
        cbs = []
        try:
            from transformers import TrainerCallback
            class _Metric(TrainerCallback):
                def __init__(self, t): self._t = t
                def on_log(self, args, state, control, logs=None, **kw):
                    if not self._t or not logs: return
                    rec = {"step": int(getattr(state, "global_step", 0))}
                    for k, v in (logs or {}).items():
                        if isinstance(v, (int, float)): rec[k] = float(v)
                    if len(rec) > 1:
                        try: self._t.logs.append("[METRIC]" + json.dumps(rec, default=str))
                        except Exception: pass
                    if self._t and state.max_steps > 0:
                        pct = min(35 + 60 * state.global_step / state.max_steps, 95)
                        loss_str = f"{logs.get('loss', logs.get('train_loss', '?'))}"
                        _safe_update(self._t, pct, f"Step {state.global_step}/{state.max_steps} | loss={loss_str}")
            cbs.append(_Metric(task))
        except Exception:
            pass

        # ---- 训练稳定性守护 ----
        try:
            from transformers import TrainerCallback as _TC
            class _StabilityGuardian(_TC):
                """监控 Loss 突刺、梯度爆炸、训练卡死

                检测逻辑:
                - Loss 突刺: 当前 loss > 滑动平均 × spike_threshold → 警告
                - 梯度爆炸: grad_norm > 10.0 → 警告
                - 连续 NaN: loss 连续 3 次 NaN/Inf → 建议降低学习率
                - Loss 不下降: 连续 50 步无下降 → 提示可能过拟合或学习率过低
                """
                def __init__(self, t, spike_threshold=3.0):
                    self._t = t
                    self._spike_th = spike_threshold
                    self._losses = []       # 滑动窗口
                    self._nan_streak = 0
                    self._best_loss = float("inf")
                    self._best_step = 0
                    self._warned_spike = False
                    self._warned_plateau = False

                def on_log(self, args, state, control, logs=None, **kw):
                    if not logs or not self._t:
                        return
                    loss = logs.get("loss") or logs.get("train_loss")
                    step = getattr(state, "global_step", 0)

                    if loss is None:
                        return

                    # NaN/Inf 检测
                    try:
                        loss = float(loss)
                        import math
                        if math.isnan(loss) or math.isinf(loss):
                            self._nan_streak += 1
                            if self._nan_streak >= 3:
                                self._t.logs.append(
                                    f"🚨 [稳定性警告] Step {step}: 连续 {self._nan_streak} 次 NaN/Inf loss！"
                                    "建议: 降低学习率或启用梯度裁剪")
                            return
                    except (ValueError, TypeError):
                        return

                    self._nan_streak = 0

                    # Loss 突刺检测
                    if len(self._losses) >= 5:
                        avg = sum(self._losses[-10:]) / len(self._losses[-10:])
                        if avg > 0 and loss > avg * self._spike_th and not self._warned_spike:
                            self._t.logs.append(
                                f"⚠️ [稳定性警告] Step {step}: Loss 突刺 {loss:.4f} (均值 {avg:.4f}, {loss/avg:.1f}x)")
                            self._warned_spike = True

                    self._losses.append(loss)

                    # 更新最佳
                    if loss < self._best_loss:
                        self._best_loss = loss
                        self._best_step = step
                        self._warned_spike = False  # 恢复正常后重置
                        self._warned_plateau = False

                    # 平台检测（连续 50 步无下降）
                    if step - self._best_step > 50 and not self._warned_plateau:
                        self._t.logs.append(
                            f"💡 [稳定性提示] Step {step}: 已 {step-self._best_step} 步未刷新最佳 loss "
                            f"(最佳: {self._best_loss:.4f} @ Step {self._best_step})")
                        self._warned_plateau = True

                    # 梯度范数检测
                    grad_norm = logs.get("grad_norm")
                    if grad_norm is not None:
                        try:
                            gn = float(grad_norm)
                            if gn > 10.0:
                                self._t.logs.append(
                                    f"⚠️ [梯度警告] Step {step}: grad_norm={gn:.2f} (偏大，可能不稳定)")
                        except (ValueError, TypeError):
                            pass

            cbs.append(_StabilityGuardian(task))
        except Exception:
            pass

        # EarlyStopping 只在有验证集时启用（否则会 crash）
        if has_eval:
            try:
                from transformers import EarlyStoppingCallback
                cbs.append(EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001))
            except Exception:
                pass
        return cbs

    # ================================================================
    #  SFT — 核心质量引擎
    # ================================================================

    def _train_sft(self, base_model: str, dataset_path: Any, p: Dict[str, Any],
                   task=None, backend: str = "trl"):
        _require_deps("datasets", "transformers")
        from datasets import load_dataset

        name = (p.get("output_name") or p.get("name") or "lora").strip()
        out_dir = Path(LORAS_DIR) / name
        out_dir.mkdir(parents=True, exist_ok=True)

        ds_paths = self._resolve_dataset_paths(dataset_path)

        _safe_update(task, 5, "加载模型 & tokenizer...")
        model, tok = self._load_model_tok(base_model, {**p, "_task": task})

        _safe_update(task, 20, "加载数据集...")
        ds = self._load_dataset_robust(ds_paths, task)
        if len(ds) == 0:
            raise RuntimeError("数据集为空。")
        keys = _first_keys(ds)
        if not keys:
            raise RuntimeError("数据集无法读取。")

        is_chat = "messages" in keys or "conversations" in keys
        is_alpaca = "instruction" in keys and "output" in keys
        _safe_update(task, 22,
            f"格式: {'对话' if is_chat else '指令' if is_alpaca else '纯文本'} | {len(ds)} 条")

        # ---- 训练前自动数据清洗（去重 + 空答案过滤）----
        # 数据质量是模型质量的第一决定因素
        auto_clean = bool(p.get("auto_clean", True))
        if auto_clean and len(ds) > 10:
            try:
                _safe_update(task, 23, "🧹 训练前数据清洗（去重+空答案过滤）...")
                _orig_len = len(ds)

                # 精确去重（基于 hash）
                _seen_hashes = set()
                import hashlib as _hl
                def _dedup_filter(example):
                    # 提取关键文本用于去重
                    text_parts = []
                    for k in ("instruction", "prompt", "question", "text"):
                        v = example.get(k)
                        if v and isinstance(v, str):
                            text_parts.append(v.strip())
                            break
                    if not text_parts:
                        msgs = example.get("messages") or example.get("conversations")
                        if isinstance(msgs, (list, str)):
                            text_parts.append(str(msgs)[:500])
                    h = _hl.md5("".join(text_parts).lower().encode("utf-8")).hexdigest()
                    if h in _seen_hashes:
                        return False
                    _seen_hashes.add(h)
                    return True

                ds = ds.filter(_dedup_filter, desc="去重")
                _dedup_removed = _orig_len - len(ds)

                # 空答案过滤
                _pre_filter_len = len(ds)
                def _nonempty_filter(example):
                    for k in ("output", "response", "answer", "completion"):
                        v = example.get(k)
                        if v and isinstance(v, str) and len(v.strip()) >= 3:
                            return True
                    msgs = example.get("messages") or example.get("conversations")
                    if isinstance(msgs, list):
                        for m in reversed(msgs):
                            if isinstance(m, dict) and m.get("role") in ("assistant", "gpt"):
                                c = str(m.get("content") or m.get("value") or "")
                                if len(c.strip()) >= 3:
                                    return True
                    # 对于纯 text 格式或无法判断的格式，保留
                    if "text" in (example.keys() if hasattr(example, "keys") else []):
                        return bool(example.get("text", "").strip())
                    return True

                ds = ds.filter(_nonempty_filter, desc="过滤空答案")
                _empty_removed = _pre_filter_len - len(ds)

                _total_cleaned = _dedup_removed + _empty_removed
                if _total_cleaned > 0:
                    _safe_update(task, 24,
                        f"🧹 数据清洗: {_orig_len}→{len(ds)} 条 "
                        f"(去重 {_dedup_removed}, 空答案 {_empty_removed})")
                    log(f"训练前自动清洗: {_orig_len}→{len(ds)} (去重={_dedup_removed}, 空={_empty_removed})")
                else:
                    _safe_update(task, 24, f"✅ 数据质量良好，无需清洗 ({len(ds)} 条)")
            except Exception as e:
                log(f"训练前清洗失败（不影响训练）: {e}")

        # ---- 格式化为 text 列 ----
        def _fmt(ex):
            if hasattr(ex, "to_dict"): ex = ex.to_dict()
            elif hasattr(ex, "keys") and not isinstance(ex, dict):
                ex = {k: ex[k] for k in list(ex.keys())}

            # 解析 messages 字段（可能是 list、JSON string、或 None）
            msgs = ex.get("messages")
            if isinstance(msgs, str):
                try: msgs = json.loads(msgs)
                except Exception: msgs = None
            if isinstance(msgs, list) and len(msgs) > 0:
                # 确保每个元素是 dict
                msgs = [dict(m) if hasattr(m, "keys") else m for m in msgs if m]
                msgs = [m for m in msgs if isinstance(m, dict)]
                if msgs:
                    try: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                    except Exception:
                        return "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in msgs)

            # 解析 conversations 字段
            convs = ex.get("conversations")
            if isinstance(convs, str):
                try: convs = json.loads(convs)
                except Exception: convs = None
            if isinstance(convs, list) and len(convs) > 0:
                convs = [dict(m) if hasattr(m, "keys") else m for m in convs if m]
                convs = [m for m in convs if isinstance(m, dict)]
                if convs:
                    mapped = []
                    for m in convs:
                        role = m.get("role") or m.get("from") or "user"
                        content = m.get("content") or m.get("value") or ""
                        if role in ("human", "user"): role = "user"
                        elif role in ("gpt", "assistant"): role = "assistant"
                        mapped.append({"role": role, "content": str(content)})
                    try: return tok.apply_chat_template(mapped, tokenize=False, add_generation_prompt=False)
                    except Exception:
                        return "\n".join(f"{m['role']}: {m['content']}" for m in mapped)

            if "instruction" in ex and "output" in ex:
                ins, inp, out = str(ex.get("instruction") or ""), str(ex.get("input") or ""), str(ex.get("output") or "")
                if inp: return f"### Instruction\n{ins}\n\n### Input\n{inp}\n\n### Response\n{out}"
                return f"### Instruction\n{ins}\n\n### Response\n{out}"

            if "text" in ex and ex["text"] is not None:
                return str(ex["text"])

            # 最后兜底：prompt+response / question+answer 等常见组合
            for q_key in ("prompt", "question", "input", "query"):
                for a_key in ("response", "answer", "output", "completion"):
                    if q_key in ex and a_key in ex and ex.get(q_key) and ex.get(a_key):
                        return f"### Instruction\n{ex[q_key]}\n\n### Response\n{ex[a_key]}"

            return json.dumps({k: v for k, v in ex.items() if v}, ensure_ascii=False, default=str)

        _safe_update(task, 25, "格式化数据集...")
        def _map_batch(examples):
            fk = next(iter(examples.keys()))
            return {"text": [_fmt({k: examples[k][i] for k in examples}) for i in range(len(examples[fk]))]}

        ds = ds.map(_map_batch, batched=True, batch_size=200,
                    remove_columns=[c for c in ds.column_names if c != "text"], desc="格式化")
        ds = ds.filter(lambda x: bool(x.get("text", "").strip()))
        if len(ds) == 0:
            raise RuntimeError("格式化后数据集为空。")

        # ---- 验证集 ----
        eval_ds = None
        if len(ds) > 100:
            split = ds.train_test_split(test_size=min(float(p.get("eval_ratio", 0.05)), 0.1), seed=42)
            ds, eval_ds = split["train"], split["test"]
            _safe_update(task, 28, f"训练: {len(ds)} | 验证: {len(eval_ds)}")
        else:
            _safe_update(task, 28, f"训练: {len(ds)} 条")

        max_seq = int(p.get("max_seq_len", p.get("max_seq_length", 2048)))

        # ---- Completion-only loss ----
        # 注意：DataCollatorForCompletionOnlyLM 需要 input_ids，
        # 它会把 response 部分之前的 labels 设为 -100。
        # 我们手动 pre-tokenize 后 labels = input_ids，
        # DataCollatorForCompletionOnlyLM 会在 collate 时修改 labels。
        data_collator = None
        _completion_template = None
        if is_alpaca and not is_chat:
            _completion_template = "### Response\n"
        elif is_chat:
            # 检测 tokenizer 的 assistant 标记
            # 策略：先用硬编码列表，再验证实际数据中是否存在

            # 取一条样本的格式化文本用于验证
            _sample_text = None
            try:
                if len(ds) > 0 and "text" in ds.column_names:
                    _sample_text = str(ds[0]["text"])
            except Exception:
                pass

            response_markers = [
                "<|im_start|>assistant\n",                               # Qwen / ChatML
                "<|start_header_id|>assistant<|end_header_id|>\n\n",     # Llama 3
                "<|assistant|>\n",                                        # Phi-3
                "### Assistant:\n",                                       # Vicuna
                "assistant\n",                                            # 通用 fallback
                "ASSISTANT:",                                             # 老版 Vicuna
            ]
            for marker in response_markers:
                # 验证 1: marker 能被 tokenizer 编码为合理长度的 token 序列
                ids = tok.encode(marker, add_special_tokens=False)
                if len(ids) == 0 or len(ids) >= 10:
                    continue
                # 验证 2: marker 确实出现在实际格式化后的文本中
                if _sample_text and marker not in _sample_text:
                    continue
                _completion_template = ids
                log(f"Completion-only: 匹配模板 '{marker.strip()}' → {ids}")
                break

            if _completion_template is None and _sample_text:
                # 动态检测：从 apply_chat_template 输出中寻找 "assistant" 附近的文本
                _assistant_pos = _sample_text.lower().find("assistant")
                if _assistant_pos >= 0:
                    # 向后找到换行符，取 assistant 标记到换行的部分
                    _line_end = _sample_text.find("\n", _assistant_pos)
                    if _line_end > _assistant_pos:
                        _dynamic_marker = _sample_text[_assistant_pos:_line_end + 1]
                        _dyn_ids = tok.encode(_dynamic_marker, add_special_tokens=False)
                        if 0 < len(_dyn_ids) < 10:
                            _completion_template = _dyn_ids
                            log(f"Completion-only: 动态检测模板 '{_dynamic_marker.strip()}' → {_dyn_ids}")

                if _completion_template is None:
                    log("⚠️ Completion-only: 未能匹配 assistant 标记，将使用全序列 loss")

        # ---- NEFTune (噪声嵌入) ----
        # 标准 Trainer 不支持 neftune_noise_alpha，需要手动实现
        # NEFTune: 在 embedding 层输出上加随机噪声，提升泛化 (+25% MT-Bench)
        nef_alpha = float(p.get("neftune_noise_alpha", 5.0))
        if nef_alpha > 0:
            try:
                _emb = None
                _base = model
                # 找到最底层的实际模型
                for attr in ("base_model.model", "base_model", "model"):
                    try:
                        _inner = model
                        for part in attr.split("."):
                            _inner = getattr(_inner, part)
                        if hasattr(_inner, "get_input_embeddings"):
                            _base = _inner
                            break
                    except AttributeError:
                        continue
                _emb = _base.get_input_embeddings()
                if _emb is not None:
                    import torch
                    _orig_emb_forward = _emb.forward
                    def _neftune_forward(x):
                        out = _orig_emb_forward(x)
                        if model.training:
                            dims = torch.tensor(out.size(1) * out.size(2), dtype=out.dtype, device=out.device)
                            mag = nef_alpha / torch.sqrt(dims)
                            out = out + torch.zeros_like(out).uniform_(-1, 1) * mag
                        return out
                    _emb.forward = _neftune_forward
                    _safe_update(task, 31, f"✅ NEFTune α={nef_alpha}")
            except Exception as e:
                log(f"NEFTune 初始化失败（不影响训练）: {e}")

        # ---- SFTConfig or fallback ----
        common = self._common_args(out_dir, p)
        cbs = self._callbacks(task, has_eval=eval_ds is not None)

        # 如果有验证集，确保 common args 包含 eval 相关配置
        # (EarlyStoppingCallback 要求 load_best_model_at_end=True)
        if eval_ds is not None:
            try:
                from transformers import TrainingArguments as _TA
                import inspect as _ins
                _ta_params = set(_ins.signature(_TA.__init__).parameters.keys())
                if "eval_strategy" in _ta_params:
                    common["eval_strategy"] = "steps"
                elif "evaluation_strategy" in _ta_params:
                    common["evaluation_strategy"] = "steps"
                common["eval_steps"] = max(common.get("logging_steps", 5) * 5, 50)
                if "load_best_model_at_end" in _ta_params:
                    common["load_best_model_at_end"] = True
                    common["metric_for_best_model"] = "eval_loss"
                    common["greater_is_better"] = False
            except Exception:
                # 如果检测失败，不加 eval config，也去掉 EarlyStopping
                cbs = [c for c in cbs if type(c).__name__ != "EarlyStoppingCallback"]

        # ---- Tokenize 数据集 ----
        _safe_update(task, 32, "Tokenize 数据集...")
        _tok_cols = {"input_ids", "attention_mask", "labels"}

        # 检测模型是否需要 token_type_ids（Gemma 3 等模型训练时强制要求）
        _needs_token_type_ids = False
        try:
            # 穿透 PeftModel/LoRA 包装，找到真实模型的 config
            _cfg = None
            for _attr_chain in ("config", "base_model.model.config", "base_model.config",
                                "model.config", "model.model.config"):
                _obj = model
                try:
                    for _part in _attr_chain.split("."):
                        _obj = getattr(_obj, _part)
                    _cfg = _obj
                    break
                except AttributeError:
                    continue

            if _cfg is not None:
                _model_type = getattr(_cfg, "model_type", "").lower()
                # Gemma 3 所有变体（gemma3）训练时强制要求 token_type_ids
                if "gemma3" in _model_type:
                    _needs_token_type_ids = True
                    log("检测到 Gemma 3 模型，将生成 token_type_ids（纯文本=全0）")
        except Exception:
            pass

        if _needs_token_type_ids:
            _tok_cols.add("token_type_ids")

        def _pretokenize(examples):
            # 确保所有 text 都是字符串
            texts = [str(t) if t is not None else "" for t in examples["text"]]
            out = tok(texts, truncation=True, max_length=max_seq, padding=False,
                      return_tensors=None)
            out["labels"] = [ids[:] for ids in out["input_ids"]]
            # Gemma 3 等模型需要 token_type_ids（纯文本训练时全填 0）
            if _needs_token_type_ids and "token_type_ids" not in out:
                out["token_type_ids"] = [[0] * len(ids) for ids in out["input_ids"]]
            return out

        ds = ds.map(_pretokenize, batched=True, batch_size=200,
                    remove_columns=ds.column_names, desc="Tokenize")
        # 安全清理：确保只剩 tokenizer 输出列
        _extra = [c for c in ds.column_names if c not in _tok_cols]
        if _extra:
            ds = ds.remove_columns(_extra)

        if eval_ds is not None:
            eval_ds = eval_ds.map(_pretokenize, batched=True, batch_size=200,
                                  remove_columns=eval_ds.column_names, desc="Tokenize eval")
            _extra_e = [c for c in eval_ds.column_names if c not in _tok_cols]
            if _extra_e:
                eval_ds = eval_ds.remove_columns(_extra_e)

        # 安全检查
        if "input_ids" not in ds.column_names:
            raise RuntimeError(
                f"Tokenize 失败：数据集列 = {ds.column_names}。"
                "请检查数据集格式是否正确。"
            )
        log(f"Tokenize 完成: {len(ds)} 条, 列 = {ds.column_names}")

        # ---- Sample Packing（短样本拼接 → 训练效率翻 2-3 倍）----
        # 原理: 把多条短样本拼进同一个 max_seq_len 序列，减少 padding 浪费
        # 对于平均长度 << max_seq_len 的数据集效果尤为显著
        #
        # ⚠️ 注意：Packing 与 Completion-only loss 不兼容！
        # DataCollatorForCompletionOnlyLM 只能找到每个序列的第一个 response template，
        # 打包后一个序列含多个样本 → 后续样本的 instruction 部分不会被 mask → loss 计算错误。
        use_packing = bool(p.get("use_packing", True))
        _packing_applied = False
        _ds_for_comparison = ds  # 保存打包前的原始数据集用于训练后质量对比
        if use_packing and _completion_template is not None:
            _safe_update(task, 33,
                "📦 Packing 跳过: 与 Completion-only loss 不兼容（优先保证 loss 质量）")
            use_packing = False
        if use_packing and len(ds) > 20:
            try:
                _safe_update(task, 33, "📦 Sample Packing（拼接短样本提效）...")

                # 统计当前 padding 浪费率
                _sample_lens = []
                for i in range(min(200, len(ds))):
                    _sample_lens.append(len(ds[i]["input_ids"]))
                _avg_len = sum(_sample_lens) / len(_sample_lens)
                _waste_ratio = 1 - (_avg_len / max_seq)

                if _waste_ratio > 0.3:  # padding 浪费 > 30% 才启用
                    import torch as _torch
                    _eos_id = tok.eos_token_id or 0

                    def _pack_samples(ds_in, max_len):
                        """将多条短序列拼接成满序列"""
                        packed_input_ids = []
                        packed_attention = []
                        packed_labels = []
                        packed_ttids = [] if _needs_token_type_ids else None

                        buf_ids, buf_attn, buf_labels, buf_ttids = [], [], [], []

                        for i in range(len(ds_in)):
                            ids = ds_in[i]["input_ids"]
                            attn = ds_in[i]["attention_mask"]
                            lbls = ds_in[i]["labels"]
                            ttids = ds_in[i].get("token_type_ids") if _needs_token_type_ids else None

                            # 如果单条已经 >= max_len，直接作为一条
                            if len(ids) >= max_len:
                                packed_input_ids.append(ids[:max_len])
                                packed_attention.append(attn[:max_len])
                                packed_labels.append(lbls[:max_len])
                                if packed_ttids is not None:
                                    packed_ttids.append((ttids or [0]*len(ids))[:max_len])
                                continue

                            # 如果 buffer + 当前样本超过 max_len，先保存 buffer
                            if buf_ids and len(buf_ids) + len(ids) + 1 > max_len:
                                # pad to max_len
                                pad_len = max_len - len(buf_ids)
                                packed_input_ids.append(buf_ids + [tok.pad_token_id or 0] * pad_len)
                                packed_attention.append(buf_attn + [0] * pad_len)
                                packed_labels.append(buf_labels + [-100] * pad_len)
                                if packed_ttids is not None:
                                    packed_ttids.append(buf_ttids + [0] * pad_len)
                                buf_ids, buf_attn, buf_labels, buf_ttids = [], [], [], []

                            # 加入 buffer（用 EOS 分隔不同样本）
                            if buf_ids:
                                buf_ids.append(_eos_id)
                                buf_attn.append(1)
                                buf_labels.append(-100)  # 分隔符不计 loss
                                if _needs_token_type_ids:
                                    buf_ttids.append(0)

                            buf_ids.extend(ids)
                            buf_attn.extend(attn)
                            buf_labels.extend(lbls)
                            if _needs_token_type_ids:
                                buf_ttids.extend(ttids or [0]*len(ids))

                        # 保存最后的 buffer
                        if buf_ids:
                            pad_len = max_len - len(buf_ids)
                            if pad_len >= 0:
                                packed_input_ids.append(buf_ids + [tok.pad_token_id or 0] * pad_len)
                                packed_attention.append(buf_attn + [0] * pad_len)
                                packed_labels.append(buf_labels + [-100] * pad_len)
                                if packed_ttids is not None:
                                    packed_ttids.append(buf_ttids + [0] * pad_len)
                            else:
                                packed_input_ids.append(buf_ids[:max_len])
                                packed_attention.append(buf_attn[:max_len])
                                packed_labels.append(buf_labels[:max_len])
                                if packed_ttids is not None:
                                    packed_ttids.append(buf_ttids[:max_len])

                        return packed_input_ids, packed_attention, packed_labels, packed_ttids

                    _packed_ids, _packed_attn, _packed_lbls, _packed_ttids = _pack_samples(ds, max_seq)
                    _orig_count = len(ds)

                    if _packed_ids:
                        from datasets import Dataset as _DS
                        _pack_dict = {
                            "input_ids": _packed_ids,
                            "attention_mask": _packed_attn,
                            "labels": _packed_lbls,
                        }
                        if _packed_ttids is not None:
                            _pack_dict["token_type_ids"] = _packed_ttids
                        ds = _DS.from_dict(_pack_dict)
                        _pack_ratio = _orig_count / max(len(ds), 1)
                        _safe_update(task, 34,
                            f"📦 Packing: {_orig_count}→{len(ds)} 条 "
                            f"(效率 {_pack_ratio:.1f}x, 原浪费率 {_waste_ratio:.0%})")
                        log(f"Sample Packing: {_orig_count}→{len(ds)} ({_pack_ratio:.1f}x)")
                        _packing_applied = True
                    else:
                        _safe_update(task, 34, "📦 Packing: 无需（样本已足够长）")
                else:
                    _safe_update(task, 34,
                        f"📦 Packing 跳过: 平均长度 {_avg_len:.0f}/{max_seq} (浪费率 {_waste_ratio:.0%} < 30%)")
            except Exception as e:
                log(f"Sample Packing 失败（不影响训练）: {e}")

        # ---- 构建 Trainer ----
        # ⚠️ 关键：确保 Trainer 没有被 Unsloth patch。
        # 使用轻量级检查（不删除 modeling 模块，避免破坏已加载的模型）
        if not p.get("_backend_unsloth"):
            _ensure_clean_trainer()

        from transformers import TrainingArguments, Trainer as HFTrainer

        # 兼容性修复：transformers 4.46+ 可能向 forward() 传入模型不支持的参数
        # （如 causal_mask），导致 Phi-3/Gemma 等模型报错。
        # Unsloth/PeftModel/accelerate 有多层包装，需要在最底层的实际模型上打补丁。
        import inspect as _inspect
        def _patch_forward(m):
            """给模型的 forward 打补丁，过滤掉不支持的参数"""
            if getattr(m, '_forgex_patched', False):
                return
            try:
                orig = m.forward
                sig = _inspect.signature(orig)
                params = sig.parameters
                param_names = set(params.keys())
                has_var_kw = any(p_.kind == _inspect.Parameter.VAR_KEYWORD for p_ in params.values())
                if has_var_kw:
                    return
                def _filtered_forward(*args, **kwargs):
                    clean_kw = {k: v for k, v in kwargs.items() if k in param_names}
                    return orig(*args, **clean_kw)
                m.forward = _filtered_forward
                m._forgex_patched = True
            except Exception:
                pass

        # 逐层找到实际的 CausalLM 模型并打补丁
        _patch_forward(model)  # 顶层（可能是 PeftModel）
        for attr in ("base_model", "model", "base_model.model"):
            _inner = model
            try:
                for part in attr.split("."):
                    _inner = getattr(_inner, part)
                if _inner is not model and hasattr(_inner, "forward"):
                    _patch_forward(_inner)
            except (AttributeError, Exception):
                pass

        args = TrainingArguments(**common)

        # 如果没有自定义 data_collator（completion-only），用 DataCollatorForSeq2Seq
        # 它会正确 pad input_ids 和 labels（labels pad 为 -100）
        # 注意：不传 model=，那是给 encoder-decoder 用的，causal LM 不需要
        _using_completion_only = False
        if _completion_template is not None:
            try:
                # trl 0.14+ 可能移动了位置
                try:
                    from trl import DataCollatorForCompletionOnlyLM
                except ImportError:
                    try:
                        from trl.trainer import DataCollatorForCompletionOnlyLM
                    except ImportError:
                        from trl.trainer.utils import DataCollatorForCompletionOnlyLM
                data_collator = DataCollatorForCompletionOnlyLM(
                    response_template=_completion_template, tokenizer=tok)
                tpl_str = _completion_template if isinstance(_completion_template, str) else f"ids={_completion_template}"
                _safe_update(task, 34, f"✅ Completion-only loss（{tpl_str[:30]}）")
                _using_completion_only = True
            except Exception as e:
                log(f"Completion-only collator 创建失败（不影响训练）: {e}")
                data_collator = None

        if not data_collator:
            from transformers import DataCollatorForSeq2Seq
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tok, padding=True, label_pad_token_id=-100)

        tkw = dict(model=model, train_dataset=ds,
                   eval_dataset=eval_ds, args=args, callbacks=cbs,
                   data_collator=data_collator,
                   **_tok_kwarg(tok, HFTrainer))

        _safe_update(task, 35,
            f"SFT | NEFTune α={nef_alpha} | "
            f"LabelSmooth={p.get('label_smoothing', 0.1)} | "
            f"cosine | seq={max_seq}")

        # MoLoRA 模式: 用自定义 Trainer (加负载均衡损失)
        _use_molora = bool(p.get("use_molora", False))
        if _use_molora:
            from core.molora import make_molora_trainer_class
            _MoLoRATrainer = make_molora_trainer_class(
                HFTrainer, aux_weight=float(p.get("molora_aux_weight", 0.01)))
            trainer = _MoLoRATrainer(**tkw)
            _safe_update(task, 36, "🧠 MoLoRA 训练模式 (含负载均衡损失)")
        else:
            trainer = HFTrainer(**tkw)

        # 训练前保存基础元信息（断点续传需要，即使训练崩溃也能恢复）
        try:
            _early_meta = {
                "base_model": base_model, "method": "sft",
                "datasets": [str(d) for d in ds_paths],
                "params": {
                    "lr": float(p.get("lr", 2e-4)),
                    "batch_size": int(p.get("batch_size", 1)),
                    "epochs": float(p.get("epochs", 1)),
                    "max_seq_len": max_seq,
                    "use_qlora": bool(p.get("use_qlora")),
                    "rank": int(p.get("rank", 64)),
                    "gradient_accumulation_steps": int(p.get("gradient_accumulation_steps", 4)),
                    "warmup_ratio": float(p.get("warmup_ratio", 0.05)),
                    "use_dora": bool(p.get("use_dora", True)),
                    "use_rslora": bool(p.get("use_rslora", True)),
                    "use_packing": bool(p.get("use_packing", True)),
                    "auto_clean": bool(p.get("auto_clean", True)),
                    "label_smoothing": float(p.get("label_smoothing", 0.1)),
                    "neftune_noise_alpha": nef_alpha,
                },
                "status": "training",
            }
            (out_dir / "forgex_meta.json").write_text(
                json.dumps(_early_meta, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass

        # Resume
        ckpt = p.get("resume_from_checkpoint")
        try:
            if ckpt and Path(ckpt).exists():
                _safe_update(task, 36, f"恢复 checkpoint: {ckpt}")
                trainer.train(resume_from_checkpoint=str(ckpt))
            else:
                trainer.train()
        except AttributeError as ae:
            # Unsloth 注入的 apply_qkv / _original_forward 等残留 patch
            ae_msg = str(ae)
            if "apply_qkv" in ae_msg or "unsloth" in ae_msg.lower() or "_original_forward" in ae_msg:
                _safe_update(task, 36, f"⚠️ 检测到 Unsloth 残留 patch ({ae_msg})，清除后重试...")
                _nuke_unsloth_patches(force=True)
                # 释放旧模型显存后再加载新模型，避免 OOM
                _cleanup_vram(trainer, model)
                del trainer
                import sys as _sys
                import importlib
                model_type = type(model).__module__
                if model_type and model_type in _sys.modules:
                    try:
                        importlib.reload(_sys.modules[model_type])
                    except Exception:
                        pass
                # 重新构建
                model_new, tok_new = self._load_model_tok(base_model, p)
                if model_new is not None:
                    _patch_forward(model_new)
                    tkw["model"] = model_new
                    tkw["tokenizer"] = tok_new
                    trainer = HFTrainer(**tkw)
                    trainer.train()
                else:
                    raise RuntimeError(f"Unsloth patch 清除后重新加载模型失败。原始错误: {ae}") from ae
            else:
                raise

        trainer.save_model(str(out_dir))
        tok.save_pretrained(str(out_dir))

        # MoLoRA: 保存专家统计 + 合并进基座 → 标准模型
        if _use_molora:
            try:
                from core.molora import (
                    save_molora_checkpoint, merge_molora_to_base,
                    get_expert_usage_summary,
                )
                _molora_cfg = getattr(model, "_molora_config", None)
                if _molora_cfg:
                    _molora_ckpt = out_dir / "molora_checkpoint"
                    save_molora_checkpoint(model, _molora_ckpt, _molora_cfg)
                    _usage = get_expert_usage_summary(model, _molora_cfg.expert_labels)
                    _safe_update(task, 92, f"🧠 {_usage}")
                    log(_usage)
                    _safe_update(task, 93, "🧠 合并专家到基座 → 标准模型...")
                    model = merge_molora_to_base(model, strategy="usage")
                    _merged_dir = out_dir / "merged_standard"
                    _merged_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(_merged_dir), safe_serialization=True)
                    tok.save_pretrained(str(_merged_dir))
                    _safe_update(task, 95,
                        f"✅ 标准模型已保存: {_merged_dir} (兼容 Ollama/vLLM)")
            except Exception as e:
                log(f"⚠️ MoLoRA 合并失败: {e}")

        # 保存元信息（完整训练配置 + 时间戳，方便复现）
        import time as _time
        train_end = _time.time()

        # 提取最终 loss（从 trainer.state）
        _final_loss = None
        _eval_loss = None
        try:
            if hasattr(trainer, "state") and hasattr(trainer.state, "log_history"):
                for entry in reversed(trainer.state.log_history):
                    if _final_loss is None and "loss" in entry:
                        _final_loss = entry["loss"]
                    if _eval_loss is None and "eval_loss" in entry:
                        _eval_loss = entry["eval_loss"]
                    if _final_loss is not None and _eval_loss is not None:
                        break
        except Exception:
            pass

        meta = {
            "base_model": base_model, "method": "sft",
            "dataset_size": len(ds), "max_seq": max_seq,
            "rank": int(p.get("rank", 64)),
            "lr": float(p.get("lr", 2e-4)),
            "batch_size": int(p.get("batch_size", 1)),
            "gradient_accumulation_steps": int(p.get("gradient_accumulation_steps", 4)),
            "epochs": float(p.get("epochs", 1)),
            "neftune": nef_alpha,
            "completion_only": _using_completion_only,
            "qlora": bool(p.get("use_qlora")),
            "unsloth": bool(p.get("_backend_unsloth")),
            "final_loss": _final_loss,
            "eval_loss": _eval_loss,
            # v3.0 新增质量特性
            "dora": bool(p.get("use_dora", True)),
            "rslora": bool(p.get("use_rslora", True)),
            "label_smoothing": float(p.get("label_smoothing", 0.1)),
            "sample_packing": _packing_applied,
            "auto_clean": bool(p.get("auto_clean", True)),
            "version": "v3.0",
            "timestamp": _time.strftime("%Y-%m-%d %H:%M:%S"),
            # 断点续传所需信息
            "datasets": [str(d) for d in ds_paths],
            "params": {
                "lr": float(p.get("lr", 2e-4)),
                "batch_size": int(p.get("batch_size", 1)),
                "epochs": float(p.get("epochs", 1)),
                "max_seq_len": max_seq,
                "use_qlora": bool(p.get("use_qlora")),
                "rank": int(p.get("rank", 64)),
                "gradient_accumulation_steps": int(p.get("gradient_accumulation_steps", 4)),
                "warmup_ratio": float(p.get("warmup_ratio", 0.05)),
                "use_dora": bool(p.get("use_dora", True)),
                "use_rslora": bool(p.get("use_rslora", True)),
                "use_packing": bool(p.get("use_packing", True)),
                "auto_clean": bool(p.get("auto_clean", True)),
                "label_smoothing": float(p.get("label_smoothing", 0.1)),
                "neftune_noise_alpha": nef_alpha,
            },
        }
        try: (out_dir / "forgex_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        except Exception: pass

        # ---- 训练后质量对比（自动 A/B Test）----
        # 从训练数据中抽取几条，用训练后的模型生成回答，与原答案对比
        try:
            _safe_update(task, 95, "🔬 训练后质量对比...")
            _comparison = self._post_train_comparison(model, tok, _ds_for_comparison, max_seq, task)
            if _comparison:
                (out_dir / "quality_comparison.json").write_text(
                    json.dumps(_comparison, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
                _n_better = sum(1 for c in _comparison if c.get("quality") == "good")
                _safe_update(task, 98,
                    f"🔬 质量对比: {_n_better}/{len(_comparison)} 条回答质量达标")
        except Exception as e:
            log(f"训练后对比失败（不影响结果）: {e}")

        _safe_update(task, 100, f"✅ 完成: {out_dir}")

        # ═══ 关键: 释放训练占用的 GPU 显存 ═══
        _cleanup_vram(trainer, model, tok, ds, eval_ds)
        del trainer, model, tok, ds, eval_ds, _ds_for_comparison
        return str(out_dir)

    def _post_train_comparison(self, model, tok, ds, max_seq, task=None):
        """训练后自动质量对比 — 抽样测试训练效果

        从训练数据中抽取 5 条，让训练后的模型生成回答，
        通过长度、关键词覆盖率等启发式指标评估质量。
        """
        import random
        import torch

        n_samples = min(5, len(ds))
        if n_samples == 0:
            return []

        # 随机抽样
        indices = random.sample(range(len(ds)), n_samples)
        comparisons = []

        model.eval()
        device = next(model.parameters()).device

        for idx in indices:
            try:
                input_ids = ds[idx]["input_ids"]
                # 取前 1/3 作为 prompt（模拟推理时的输入）
                prompt_len = max(20, len(input_ids) // 3)
                prompt_ids = input_ids[:prompt_len]

                with torch.no_grad():
                    input_tensor = torch.tensor([prompt_ids], device=device)
                    output = model.generate(
                        input_tensor,
                        max_new_tokens=min(200, max_seq - prompt_len),
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tok.pad_token_id or tok.eos_token_id,
                    )

                generated = tok.decode(output[0][prompt_len:], skip_special_tokens=True)
                original = tok.decode(input_ids[prompt_len:], skip_special_tokens=True)
                prompt_text = tok.decode(prompt_ids, skip_special_tokens=True)

                # 启发式质量评估
                quality = "good"
                notes = []

                # 检查是否生成了有意义的内容
                if len(generated.strip()) < 10:
                    quality = "poor"
                    notes.append("生成内容过短")

                # 检查是否有重复循环
                if generated and len(generated) > 50:
                    words = generated.split()
                    if len(words) > 10:
                        unique_ratio = len(set(words)) / len(words)
                        if unique_ratio < 0.3:
                            quality = "poor"
                            notes.append("检测到重复循环")

                # 检查与原答案的关键词重叠
                if original:
                    orig_words = set(original.split())
                    gen_words = set(generated.split())
                    if orig_words:
                        overlap = len(orig_words & gen_words) / len(orig_words)
                        if overlap > 0.2:
                            notes.append(f"关键词重叠 {overlap:.0%}")

                comparisons.append({
                    "prompt": prompt_text[:200],
                    "original": original[:300],
                    "generated": generated[:300],
                    "quality": quality,
                    "notes": notes,
                })
            except Exception as e:
                comparisons.append({"error": str(e), "quality": "error"})

        return comparisons

    # ================================================================
    #  DPO / ORPO
    # ================================================================

    def _train_dpo_like(self, kind, base_model, dataset_path, p, task=None, backend="trl"):
        _require_deps("datasets", "transformers", "trl")
        from datasets import load_dataset
        from trl import DPOTrainer

        ORPOTrainer = None
        if kind == "orpo":
            from trl import ORPOTrainer as _O
            ORPOTrainer = _O

        name = (p.get("output_name") or p.get("name") or f"{kind}_lora").strip()
        out_dir = Path(LORAS_DIR) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        ds_paths = self._resolve_dataset_paths(dataset_path)

        _safe_update(task, 10, "加载模型...")
        model, tok = self._load_model_tok(base_model, {**p, "_task": task})

        _safe_update(task, 25, "加载偏好数据集...")
        ds = self._load_dataset_robust(ds_paths, task)
        need = {"prompt", "chosen", "rejected"}
        if not need.issubset(_first_keys(ds)):
            raise RuntimeError(f"{kind.upper()} needs {sorted(need)}. Got: {sorted(list(_first_keys(ds)))[:20]}")

        common = self._common_args(out_dir, p)
        max_seq = int(p.get("max_seq_len", p.get("max_seq_length", 2048)))
        Cls = ORPOTrainer if kind == "orpo" else DPOTrainer

        _safe_update(task, 35, f"启动 {kind.upper()}...")
        # 确保 Unsloth patches 不干扰 DPO/ORPO
        if not p.get("_backend_unsloth"):
            _ensure_clean_trainer()
        try:
            from trl import DPOConfig
            cfg = DPOConfig(**common, max_prompt_length=max_seq // 2, max_length=max_seq,
                            beta=float(p.get("beta", 0.1)))
            trainer = Cls(model=model, **_tok_kwarg(tok, Cls), train_dataset=ds,
                          args=cfg, callbacks=self._callbacks(task))
        except (ImportError, TypeError):
            from transformers import TrainingArguments
            args = TrainingArguments(**common)
            try:
                trainer = Cls(model=model, **_tok_kwarg(tok, Cls), train_dataset=ds, args=args,
                              beta=float(p.get("beta", 0.1)),
                              max_prompt_length=max_seq // 2, max_length=max_seq)
            except TypeError:
                trainer = Cls(**_filter_kwargs_for_callable(Cls.__init__,
                    dict(model=model, **_tok_kwarg(tok, Cls), train_dataset=ds, args=args)))

        trainer.train()
        trainer.save_model(str(out_dir))
        tok.save_pretrained(str(out_dir))
        _safe_update(task, 100, f"✅ 完成: {out_dir}")
        _cleanup_vram(trainer, model, tok, ds)
        del trainer, model, tok, ds
        return str(out_dir)

    # ================================================================
    #  KTO
    # ================================================================

    def _train_kto(self, base_model, dataset_path, p, task=None, backend="trl"):
        _require_deps("datasets", "transformers", "trl")
        from datasets import load_dataset
        from trl import KTOTrainer

        name = (p.get("output_name") or p.get("name") or "kto_lora").strip()
        out_dir = Path(LORAS_DIR) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        ds_paths = self._resolve_dataset_paths(dataset_path)

        _safe_update(task, 10, "加载模型...")
        model, tok = self._load_model_tok(base_model, {**p, "_task": task})

        _safe_update(task, 25, "加载 KTO 数据集...")
        ds = self._load_dataset_robust(ds_paths, task)
        keys = _first_keys(ds)

        if {"prompt", "completion", "label"}.issubset(keys):
            pass
        elif {"prompt", "chosen", "rejected"}.issubset(keys):
            # 将 DPO 格式 {prompt, chosen, rejected} 展开为 KTO 格式
            # {prompt, completion, label}（每行变两行）
            # 必须用 batched=True，否则返回列表会变成嵌套列表列而非展开行
            def _expand_to_kto(batch):
                prompts, completions, labels = [], [], []
                for p_text, c, r in zip(batch["prompt"], batch["chosen"], batch["rejected"]):
                    prompts.extend([p_text, p_text])
                    completions.extend([c, r])
                    labels.extend([True, False])
                return {"prompt": prompts, "completion": completions, "label": labels}
            ds = ds.map(_expand_to_kto, batched=True, batch_size=500,
                        remove_columns=ds.column_names)
        else:
            raise RuntimeError(f"KTO needs {{prompt,completion,label}} or {{prompt,chosen,rejected}}. Got: {sorted(list(keys))[:20]}")

        common = self._common_args(out_dir, p)
        max_seq = int(p.get("max_seq_len", p.get("max_seq_length", 2048)))

        _safe_update(task, 35, "启动 KTO...")
        # 确保 Unsloth patches 不干扰 KTO
        if not p.get("_backend_unsloth"):
            _ensure_clean_trainer()
        try:
            from trl import KTOConfig
            cfg = KTOConfig(**common, max_length=max_seq)
            trainer = KTOTrainer(model=model, **_tok_kwarg(tok, KTOTrainer), train_dataset=ds,
                                 args=cfg, callbacks=self._callbacks(task))
        except (ImportError, TypeError):
            from transformers import TrainingArguments
            args = TrainingArguments(**common)
            try:
                trainer = KTOTrainer(model=model, **_tok_kwarg(tok, KTOTrainer), train_dataset=ds, args=args, max_length=max_seq)
            except TypeError:
                trainer = KTOTrainer(**_filter_kwargs_for_callable(KTOTrainer.__init__,
                    dict(model=model, **_tok_kwarg(tok, KTOTrainer), train_dataset=ds, args=args)))

        trainer.train()
        trainer.save_model(str(out_dir))
        tok.save_pretrained(str(out_dir))
        _safe_update(task, 100, f"✅ 完成: {out_dir}")
        _cleanup_vram(trainer, model, tok, ds)
        del trainer, model, tok, ds
        return str(out_dir)


trainer_engine = TrainerEngine()


class Trainer(TrainerEngine):
    """Backward-compatible alias."""
    pass

__all__ = ["Trainer", "TrainerEngine", "MethodSupport"]
