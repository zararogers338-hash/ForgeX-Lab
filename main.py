# main.py - ForgeX (Refactor: Forge Pipeline UI)
from __future__ import annotations

import sys
# Gradio 284+ 组件 + HuggingFace transformers 深层调用栈
# 默认 1000 太低，会在组件树遍历和模型加载时 RecursionError
sys.setrecursionlimit(10000)

from launch_guard import require_user_launch
require_user_launch()

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

# ForgeX v3.2-user - 全方位训练锻造

# ═══════════════════════════════════════════════════════════════
# 依赖兼容层: 修补第三方库之间的版本冲突
# 必须在 import gradio / transformers 之前执行
# ═══════════════════════════════════════════════════════════════
def _patch_dependency_compat():
    """修补常见的第三方库版本冲突。"""
    _fixed = []

    # ── 0. torchvision 循环导入 ──
    # 由 core/__init__.py 的 _TorchvisionWatchdog 永久监控
    # 这里只检查状态并报告
    tv = sys.modules.get("torchvision")
    if tv and getattr(tv, "__version__", "").endswith("+forgex_shim"):
        _fixed.append("torchvision(shim)")

    # ── 1. huggingface_hub: HfFolder 已在 0.23+ 中移除 ──
    # gradio < 5.0 仍在引用它
    try:
        from huggingface_hub import HfFolder  # noqa: F401
    except ImportError:
        try:
            import huggingface_hub
            class _HfFolderCompat:
                """兼容 shim: HfFolder 的功能已被 huggingface_hub 顶层 API 替代"""
                @staticmethod
                def get_token():
                    try:
                        return huggingface_hub.get_token()
                    except Exception:
                        return None
                @staticmethod
                def save_token(token):
                    try:
                        huggingface_hub.login(token=token, add_to_git_credential=False)
                    except Exception:
                        pass
                @staticmethod
                def delete_token():
                    try:
                        huggingface_hub.logout()
                    except Exception:
                        pass
            huggingface_hub.HfFolder = _HfFolderCompat
            if "huggingface_hub" in sys.modules:
                sys.modules["huggingface_hub"].HfFolder = _HfFolderCompat
            _fixed.append("huggingface_hub.HfFolder")
        except Exception:
            pass

    # ── 2. numpy 2.0: np.PINF / np.float_ 等别名被移除 ──
    try:
        import numpy as np
        for name, val in [
            ("PINF", float("inf")), ("NINF", float("-inf")),
            ("float_", getattr(np, "float64", float)),
            ("int_", getattr(np, "int64", int)),
            ("complex_", getattr(np, "complex128", complex)),
            ("object_", object), ("str_", str),
        ]:
            if not hasattr(np, name):
                setattr(np, name, val)
                _fixed.append(f"numpy.{name}")
    except ImportError:
        pass

    # ── 3. PIL/Pillow 10+: ANTIALIAS → LANCZOS ──
    try:
        from PIL import Image
        if not hasattr(Image, "ANTIALIAS"):
            Image.ANTIALIAS = Image.LANCZOS
            _fixed.append("PIL.ANTIALIAS")
    except ImportError:
        pass

    # ── 4. gradio-client API schema 兼容层 ──
    # Gradio 4.x 在启动首页时会调用 get_api_info()。某些组件的 JSON schema
    # 会包含 additionalProperties: false；gradio-client 1.3/2.0 在递归解析这个
    # bool 值时会抛出 TypeError，导致页面打不开。这里只补 API 信息解析，
    # 不改变 UI 功能和训练逻辑。
    try:
        import gradio_client.utils as _gc_utils
        _orig_schema_to_py = getattr(_gc_utils, "_json_schema_to_python_type", None)
        if _orig_schema_to_py is not None and not getattr(_orig_schema_to_py, "_forgex_bool_schema_patch", False):
            def _forgex_json_schema_to_python_type(schema, defs):
                if isinstance(schema, bool):
                    return "Any" if schema else "None"
                return _orig_schema_to_py(schema, defs)
            _forgex_json_schema_to_python_type._forgex_bool_schema_patch = True
            _gc_utils._json_schema_to_python_type = _forgex_json_schema_to_python_type
            _fixed.append("gradio_client.bool_schema")
    except Exception:
        pass


    # ── 4b. Starlette 1.x TemplateResponse 兼容层 ──
    # Gradio 4.x 仍使用 TemplateResponse(name, context) 的旧调用格式；
    # Starlette 1.x 改为 TemplateResponse(request, name)，会把 context dict
    # 当作模板名，首页直接报 TypeError: unhashable type: 'dict'。
    # 这个补丁只恢复旧调用格式，不改 ForgeX UI/训练逻辑。
    try:
        from starlette.templating import Jinja2Templates, _TemplateResponse
        _orig_template_response = getattr(Jinja2Templates, "TemplateResponse", None)
        if _orig_template_response is not None and not getattr(_orig_template_response, "_forgex_oldstyle_patch", False):
            def _forgex_template_response(self, *args, **kwargs):
                if args and isinstance(args[0], str):
                    name = args[0]
                    context = args[1] if len(args) > 1 else kwargs.pop("context", {})
                    status_code = args[2] if len(args) > 2 else kwargs.pop("status_code", 200)
                    headers = args[3] if len(args) > 3 else kwargs.pop("headers", None)
                    media_type = args[4] if len(args) > 4 else kwargs.pop("media_type", None)
                    background = args[5] if len(args) > 5 else kwargs.pop("background", None)
                    request = context.get("request") if isinstance(context, dict) else None
                    if request is None:
                        request = kwargs.pop("request", None)
                    if request is None:
                        raise ValueError('context must include a "request" key')
                    context.setdefault("request", request)
                    for processor in getattr(self, "context_processors", []):
                        context.update(processor(request))
                    template = self.get_template(name)
                    return _TemplateResponse(
                        template,
                        context,
                        status_code=status_code,
                        headers=headers,
                        media_type=media_type,
                        background=background,
                    )
                return _orig_template_response(self, *args, **kwargs)
            _forgex_template_response._forgex_oldstyle_patch = True
            Jinja2Templates.TemplateResponse = _forgex_template_response
            _fixed.append("starlette.TemplateResponse")
    except Exception:
        pass

    # ── 5. transformers 兼容层 (safe_loader) ──
    try:
        from core.safe_loader import _patch_transformers_compat
        _patch_transformers_compat()
    except Exception:
        pass

    if _fixed:
        print(f"[compat] 已修补: {', '.join(_fixed)}")

_patch_dependency_compat()

import inspect
import gradio as gr


def _chatbot_messages(**kwargs):
    """Create a messages-mode Chatbot across Gradio versions.

    Gradio 4.x accepts type="messages"; Gradio 6.x removed that keyword and
    uses message dictionaries by default. Keeping this wrapper lets the UI open
    whether the user's machine has the pinned 4.x runtime or a newer Gradio.
    """
    if "type" in inspect.signature(gr.Chatbot).parameters:
        kwargs.setdefault("type", "messages")
    return gr.Chatbot(**kwargs)

from core.logger import log
from core.config import config, DATASETS_DIR, LORAS_DIR, MODELS_CACHE_DIR, LOGS_DIR, PROJECT_DIR
from core.dataset_manager import DatasetManager
from core.task_queue import task_queue, TaskStatus
from core.trainer import Trainer
from core.merger import Merger, native_merger
from core.exporter import Exporter
from core.benchmark import benchmark
from core.utils import safe_json_load, safe_json_save, human_size, detect_gpu
from core.model_manager import model_manager
from core.version import UI_TITLE, UI_HEADER

RECIPES_DIR = (PROJECT_DIR / "data" / "recipes")
RECIPES_DIR.mkdir(parents=True, exist_ok=True)

dm = DatasetManager()
trainer = Trainer()
merger = Merger()
exporter = Exporter()
bench = benchmark

# ---------------------------
# Helpers (带缓存，避免重复调用)
# ---------------------------

# 启动缓存：这些在 build_app 时会被多个 Tab 引用，只需计算一次
_cache_datasets = None
_cache_models = None
_cache_gpu = None
_cache_loras = None

def _get_gpu_cached():
    """GPU 信息只检测一次，缓存结果"""
    global _cache_gpu
    if _cache_gpu is None:
        _cache_gpu = detect_gpu()
    return _cache_gpu

def _list_datasets() -> List[str]:
    """列出所有数据集。直接从磁盘扫描，不依赖 index.json（防止不同步）。"""
    global _cache_datasets
    if _cache_datasets is not None:
        return _cache_datasets
    try:
        files = []
        for ext in [".jsonl", ".json", ".csv", ".txt", ".parquet"]:
            for f in sorted(DATASETS_DIR.glob(f"*{ext}")):
                if f.name == "index.json":
                    continue
                files.append(f.name)
        _cache_datasets = files
        return _cache_datasets
    except Exception:
        return []

def _invalidate_ds_cache():
    """上传/删除后清除缓存"""
    global _cache_datasets
    _cache_datasets = None

# 启动时一次性加载（后续复用缓存，不再重复读文件）
_STARTUP_DS_LIST = _list_datasets()

def _list_loras() -> List[str]:
    global _cache_loras
    if _cache_loras is not None:
        return _cache_loras
    if not LORAS_DIR.exists():
        return []
    out = []
    for p in sorted(LORAS_DIR.iterdir()):
        if not p.is_dir():
            continue
        if (p / "adapter_config.json").exists():
            out.append(p.name)
            continue
        if any((p / f).exists() for f in ["adapter_model.safetensors", "adapter_model.bin"]):
            out.append(p.name)
    _cache_loras = out
    return out

_cache_all_models = None
def _extract_model_name_from_label(label: str) -> str:
    """从下拉标签中提取模型目录名。
    例: '[LoRA] mylora ← Qwen2.5' → 'mylora'
        '[完整模型] merged_v2'   → 'merged_v2'
    """
    import re
    clean = re.sub(r'^\[.*?\]\s*', '', label)   # 去前缀
    clean = clean.split(' ← ')[0].strip()        # 去基座提示
    return clean

def _resolve_model_path(label: str) -> str:
    """下拉标签 → 模型完整路径 (统一入口)

    支持输入:
      1. "[LoRA] my_model ← base"  →  data/loras/my_model
      2. "google/gemma-3-27b-it"   →  HF cache snapshot 路径
      3. "/absolute/path/to/model"  →  直接使用
      4. HF cache 目录名 "models--google--gemma-3-27b-it" → 解析 snapshot
    """
    clean = _extract_model_name_from_label(label)

    # 1. 绝对路径或已存在的路径 → 直接返回（但要检查是否是 HF cache 目录）
    if Path(label).exists():
        resolved = _try_resolve_hf_cache_dir(Path(label))
        if resolved:
            return resolved
        return str(Path(label))

    # 2. data/loras 下的模型
    p = LORAS_DIR / clean
    if p.exists():
        return str(p)

    # 3. HF ID 格式 (org/model) → 在 models_cache 中查找
    hf_cache_name = f"models--{clean.replace('/', '--')}"
    hf_cache_dir = MODELS_CACHE_DIR / hf_cache_name
    if hf_cache_dir.exists():
        resolved = _try_resolve_hf_cache_dir(hf_cache_dir)
        if resolved:
            return resolved
        # cache 存在但没有 snapshots → 下载未完成
        raise FileNotFoundError(
            f"模型缓存目录存在但下载未完成: {hf_cache_dir}\n"
            f"  目录内容: {[x.name for x in hf_cache_dir.iterdir()]}\n"
            f"  缺少 snapshots/ 目录 → 请重新下载或使用完整模型文件夹\n"
            f"  提示: 完整的模型文件夹应直接包含 config.json, tokenizer.json 等文件"
        )

    # 4. 也检查 HF 默认缓存
    for cache_root in _get_hf_cache_roots():
        d = cache_root / hf_cache_name
        if d.exists():
            resolved = _try_resolve_hf_cache_dir(d)
            if resolved:
                return resolved

    return str(p)


def _try_resolve_hf_cache_dir(cache_dir: Path) -> str:
    """尝试从 HF cache 目录解析出实际 snapshot 路径

    HF cache 结构:
      models--org--name/
        blobs/          ← 文件内容（硬链接）
        refs/           ← 分支引用
        snapshots/      ← 每个 commit 一个完整目录
          abc123def/    ← 这个才是模型路径
            config.json
            tokenizer.json
            model-*.safetensors
    """
    # 如果目录本身就有 config.json → 已经是模型目录
    if (cache_dir / "config.json").exists():
        return str(cache_dir)

    snap_dir = cache_dir / "snapshots"
    if not snap_dir.exists():
        return ""

    # 找最新的（或唯一的）有 config.json 的 snapshot
    best = None
    best_mtime = 0
    for s in snap_dir.iterdir():
        if s.is_dir() and (s / "config.json").exists():
            mtime = s.stat().st_mtime
            if mtime > best_mtime:
                best = s
                best_mtime = mtime

    return str(best) if best else ""


def _get_hf_cache_roots():
    """获取所有可能的 HF cache 根目录"""
    roots = [MODELS_CACHE_DIR]
    try:
        hf_default = Path.home() / ".cache" / "huggingface" / "hub"
        if hf_default.exists() and hf_default != MODELS_CACHE_DIR:
            roots.append(hf_default)
        import os
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            hf_hub = Path(hf_home) / "hub"
            if hf_hub.exists() and hf_hub not in roots:
                roots.append(hf_hub)
    except Exception:
        pass
    return roots


def _detect_model_abilities(model_path: str) -> List[str]:
    """分析 config.json 检测模型能力特征。
    
    返回能力标签列表，如: ["多模态", "MoE", "长上下文", "代码"]
    """
    abilities = []
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        return ["未知"]
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return ["未知"]

    model_type = cfg.get("model_type", "").lower()
    archs = [a.lower() for a in cfg.get("architectures", [])]

    # 多模态检测
    if any(k in cfg for k in ("vision_config", "visual_config", "image_size",
                                "vision_tower", "mm_vision_tower", "visual")):
        abilities.append("多模态(视觉)")
    if any(k in cfg for k in ("audio_config", "audio_encoder", "whisper_model")):
        abilities.append("多模态(音频)")

    # MoE 检测
    if any(k in cfg for k in ("num_experts", "num_local_experts", "num_experts_per_tok")):
        n_exp = cfg.get("num_local_experts", cfg.get("num_experts", 0))
        abilities.append(f"MoE({n_exp}专家)")
    if any("moe" in a or "mixtral" in a for a in archs):
        if not any("MoE" in a for a in abilities):
            abilities.append("MoE")

    # 长上下文
    max_pos = cfg.get("max_position_embeddings", 0)
    rope_scaling = cfg.get("rope_scaling")
    if max_pos >= 32768:
        abilities.append(f"长上下文({max_pos // 1024}K)")
    elif rope_scaling:
        abilities.append("RoPE扩展")

    # 架构特征
    hidden = cfg.get("hidden_size", 0)
    layers = cfg.get("num_hidden_layers", 0)
    if hidden > 0 and layers > 0:
        param_est = hidden * hidden * layers * 12 / 1e9  # 粗估
        if param_est > 30:
            abilities.append("大参数")

    # 模型系列特征
    name_or_path = cfg.get("_name_or_path", "").lower()
    if any(kw in name_or_path for kw in ("code", "starcoder", "codegen", "deepseek-coder")):
        abilities.append("代码")
    if any(kw in name_or_path for kw in ("math", "数学")):
        abilities.append("数学")
    if any(kw in name_or_path for kw in ("r1", "reason", "思考", "o1")):
        abilities.append("推理")
    if any(kw in name_or_path for kw in ("chinese", "qwen", "chatglm", "yi-", "baichuan")):
        abilities.append("中文")

    if not abilities:
        abilities.append(f"文本生成({model_type})")
    return abilities

def _list_task_choices() -> List[str]:
    """生成任务下拉选项: 'task_xxx | 名称 | 状态 (进度%)'"""
    tasks = task_queue.get_all_tasks()  # newest first
    choices = []
    status_icons = {
        "pending": "⏳", "running": "🔄", "completed": "✅",
        "failed": "❌", "cancelled": "🚫",
    }
    for t in tasks[:50]:  # 最多显示50个
        st = t.get("status", "")
        icon = status_icons.get(st, "❓")
        prog = t.get("progress", 0)
        name = t.get("name", "")[:40]
        tid = t.get("id", "")
        choices.append(f"{icon} {name} ({prog:.0f}%) | {tid}")
    return choices

def _extract_task_id_from_choice(choice: str) -> str:
    """从下拉选项提取 task_id"""
    if not choice:
        return ""
    # 格式: "🔄 训练xxx (45%) | task_xxx_123"
    parts = choice.rsplit(" | ", 1)
    return parts[-1].strip() if len(parts) == 2 else choice.strip()

def _list_all_model_dirs() -> List[str]:
    """列出 data/loras 下所有有效目录 + HF cache 中的模型"""
    global _cache_all_models
    if _cache_all_models is not None:
        return _cache_all_models
    out = []

    # 1. data/loras 下的 LoRA 和完整模型
    if LORAS_DIR.exists():
        try:
            entries = sorted(LORAS_DIR.iterdir())
        except (PermissionError, OSError) as e:
            log(f"⚠️ 无法读取模型目录 {LORAS_DIR}: {e}")
            entries = []
        for p in entries:
            if not p.is_dir():
                continue
            is_lora = (p / "adapter_config.json").exists()
            is_model = (p / "config.json").exists()
            has_adapter_weights = any((p / f).exists() for f in ["adapter_model.safetensors", "adapter_model.bin"])
            has_model_weights = any((p / f).exists() for f in ["model.safetensors", "pytorch_model.bin"]) or \
                                any(p.glob("model-*.safetensors"))
            if is_lora or has_adapter_weights:
                base_hint = ""
                try:
                    import json as _j
                    acfg = _j.loads((p / "adapter_config.json").read_text(encoding="utf-8"))
                    base_name = acfg.get("base_model_name_or_path", "")
                    if base_name:
                        base_hint = f" ← {Path(base_name).name}"
                except Exception:
                    pass
                out.append(f"[LoRA] {p.name}{base_hint}")
            elif is_model and has_model_weights:
                out.append(f"[完整模型] {p.name}")
            elif is_model:
                out.append(f"[模型] {p.name}")

    # 2. HF cache 中已下载完成的模型
    seen_hf = set()
    for cache_root in _get_hf_cache_roots():
        try:
            import glob as _glob
            for d in _glob.glob(str(cache_root / "models--*")):
                dp = Path(d)
                hf_name = dp.name.replace("models--", "").replace("--", "/")
                if hf_name in seen_hf:
                    continue
                # 检查是否下载完成（有 snapshots + config.json）
                resolved = _try_resolve_hf_cache_dir(dp)
                if resolved:
                    seen_hf.add(hf_name)
                    out.append(f"[HF缓存] {hf_name}")
        except Exception:
            continue

    _cache_all_models = out
    return out

def _invalidate_loras_cache():
    global _cache_loras, _cache_all_models
    _cache_loras = None
    _cache_all_models = None

def _list_recipes() -> List[str]:
    return [p.name for p in sorted(RECIPES_DIR.glob("*.json"))]

def _list_local_hf_model_ids() -> List[str]:
    global _cache_models
    if _cache_models is not None:
        return _cache_models
    try:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as ex:
            fut = ex.submit(model_manager.scan_local_models)
            items = fut.result(timeout=15)  # 15 秒超时保护
        ids = sorted({it.get("name","") for it in items if it.get("source") == "huggingface" and it.get("name")})
        _cache_models = ids
        return ids
    except Exception:
        return []

def _refresh_local_hf_model_ids():
    """强制刷新缓存"""
    global _cache_models
    _cache_models = None
    try:
        ids = _list_local_hf_model_ids()
    except Exception:
        ids = []
    if not ids:
        return gr.update(choices=["（未找到本地模型，请在 HF ID 框手动输入）"], value=None)
    return gr.update(choices=ids, value=ids[0] if len(ids) == 1 else None)

# 启动时用空列表，避免扫描 HF cache（可能 > 10 秒）
_EMPTY_MODELS = []

def _save_recipe(name: str, payload: Dict[str, Any]) -> str:
    name = (name or "").strip() or f"recipe_{int(time.time())}.json"
    if not name.endswith(".json"):
        name += ".json"
    path = RECIPES_DIR / Path(name).name
    safe_json_save(path, payload)
    return path.name

def _load_recipe(name: str) -> Dict[str, Any]:
    if not name:
        return {}
    path = RECIPES_DIR / name
    return safe_json_load(path, {})

def _poll_task(task_id: str) -> Dict[str, Any]:
    t = task_queue.get_task(task_id)
    if not t:
        return {"status": "missing", "progress": 0, "message": "task not found", "logs": []}
    return {
        "status": t.status.value,
        "progress": t.progress,
        "message": t.message,
        "error": t.error,
        "logs": t.logs[-200:],  # last 200 lines
    }

def _metrics_from_logs(log_lines: List[str]) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    for ln in (log_lines or []):
        if not isinstance(ln, str):
            continue
        if ln.startswith("[METRIC]"):
            payload = ln[len("[METRIC]"):]
            try:
                obj = json.loads(payload)
                if isinstance(obj, dict):
                    recs.append(obj)
            except Exception:
                continue
    return recs

def _make_loss_figure(log_lines: List[str]):
    """多功能图表 — 纯 HTML/CSS/SVG，不依赖 matplotlib 也不依赖 Chart.js
    自动检测训练 Loss 和考试成绩并分面板展示"""
    recs = _metrics_from_logs(log_lines)
    if not recs:
        return _empty_figure("等待数据中…（自动刷新 3s）")

    # ── 分类指标 ──
    train_pts = []          # {step, loss, eval_loss?, lr?}
    exam_questions = []     # type=exam_q
    exam_rounds_data = []   # type=exam_round

    for r in recs:
        mtype = r.get("type", "")
        if mtype == "exam_q":
            exam_questions.append(r)
        elif mtype == "exam_round":
            exam_rounds_data.append(r)
        elif mtype == "exam_q_student":
            pass  # 学生做题指标，不用于图表
        else:
            step = r.get("step")
            loss = r.get("loss") or r.get("train_loss")
            if step is not None and loss is not None:
                try:
                    pt = {"step": float(step), "loss": float(loss)}
                    el = r.get("eval_loss")
                    if el is not None:
                        pt["eval_loss"] = float(el)
                    lr = r.get("learning_rate")
                    if lr is not None:
                        pt["lr"] = float(lr)
                    train_pts.append(pt)
                except (ValueError, TypeError):
                    pass

    has_train = bool(train_pts)
    has_exam = bool(exam_questions)
    has_rounds = bool(exam_rounds_data)

    if not has_train and not has_exam and not has_rounds:
        return _empty_figure("等待数据中…")

    panels_html = []

    # ══════════════════════════════════════
    #  Panel A: 训练 Loss (SVG 折线图)
    # ══════════════════════════════════════
    if has_train:
        latest_loss = train_pts[-1]["loss"]
        latest_eval = None
        for p in reversed(train_pts):
            if "eval_loss" in p:
                latest_eval = p["eval_loss"]
                break

        # SVG 尺寸
        W, H = 700, 180
        PAD_L, PAD_R, PAD_T, PAD_B = 50, 20, 10, 30
        pw, ph = W - PAD_L - PAD_R, H - PAD_T - PAD_B

        steps = [p["step"] for p in train_pts]
        losses = [p["loss"] for p in train_pts]
        s_min, s_max = min(steps), max(steps)
        l_min, l_max = min(losses) * 0.9, max(losses) * 1.05
        if s_max == s_min: s_max = s_min + 1
        if l_max == l_min: l_max = l_min + 0.1

        def _sx(s): return PAD_L + (s - s_min) / (s_max - s_min) * pw
        def _sy(l): return PAD_T + (1 - (l - l_min) / (l_max - l_min)) * ph

        # Train loss polyline
        pts_str = " ".join(f"{_sx(s):.1f},{_sy(l):.1f}" for s, l in zip(steps, losses))

        # Eval loss dots
        eval_dots = ""
        eval_pts = [(p["step"], p["eval_loss"]) for p in train_pts if "eval_loss" in p]
        for es, el in eval_pts:
            eval_dots += f'<circle cx="{_sx(es):.1f}" cy="{_sy(el):.1f}" r="3" fill="#f85149" />'

        # Y-axis labels
        y_labels = ""
        for i in range(5):
            v = l_min + (l_max - l_min) * i / 4
            y = _sy(v)
            y_labels += f'<text x="{PAD_L-5}" y="{y:.0f}" text-anchor="end" font-size="10" fill="#8b949e">{v:.3f}</text>'
            y_labels += f'<line x1="{PAD_L}" y1="{y:.0f}" x2="{W-PAD_R}" y2="{y:.0f}" stroke="#21262d" />'

        # X-axis labels
        x_labels = ""
        for i in range(min(5, len(steps))):
            idx = int(i * (len(steps) - 1) / max(4, 1))
            s = steps[idx]
            x_labels += f'<text x="{_sx(s):.0f}" y="{H-5}" text-anchor="middle" font-size="10" fill="#8b949e">{int(s)}</text>'

        header_vals = f'Loss: {latest_loss:.4f}'
        if latest_eval is not None:
            header_vals += f' | Eval: {latest_eval:.4f}'
        header_vals += f' | Steps: {int(steps[-1])}'

        panels_html.append(f'''
        <div style="margin-bottom:12px">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
            <span style="font-size:13px;font-weight:600;color:#c9d1d9">📉 训练曲线</span>
            <span style="font-size:11px;color:#58a6ff;font-family:monospace">{header_vals}</span>
          </div>
          <svg viewBox="0 0 {W} {H}" style="width:100%;background:#161b22;border-radius:6px">
            {y_labels}{x_labels}
            <polyline points="{pts_str}" fill="none" stroke="#58a6ff" stroke-width="1.5" opacity="0.9"/>
            {eval_dots}
          </svg>
          <div style="font-size:10px;color:#8b949e;margin-top:4px">
            <span style="color:#58a6ff">━</span> Train Loss
            {"<span style='margin-left:12px;color:#f85149'>●</span> Eval Loss" if eval_pts else ""}
          </div>
        </div>''')

    # ══════════════════════════════════════
    #  Panel B: 考试成绩 (CSS 柱状图 + SVG 均分线)
    # ══════════════════════════════════════
    if has_exam or has_rounds:
        q_scores = [q.get("score", 0) for q in exam_questions]
        total_q = len(q_scores)

        if total_q > 0:
            # 统计
            cum = 0
            avgs = []
            for s in q_scores:
                cum += s
                avgs.append(cum / len(avgs + [1]))  # placeholder
            # 重算
            cum2, avgs2 = 0, []
            for i, s in enumerate(q_scores):
                cum2 += s
                avgs2.append(cum2 / (i + 1))

            final_avg = avgs2[-1] if avgs2 else 0
            correct = sum(1 for s in q_scores if s >= 4)
            wrong = sum(1 for s in q_scores if s < 4)

            # ── 统计卡片 ──
            stat_cards = f'''
            <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px">
              <div style="flex:1;min-width:65px;text-align:center;background:#161b22;border-radius:6px;padding:6px">
                <div style="font-size:1.3em;font-weight:700;color:#58a6ff">{final_avg:.1f}</div>
                <div style="font-size:0.7em;color:#8b949e">均分</div>
              </div>
              <div style="flex:1;min-width:65px;text-align:center;background:#161b22;border-radius:6px;padding:6px">
                <div style="font-size:1.3em;font-weight:700;color:#3fb950">{correct}</div>
                <div style="font-size:0.7em;color:#8b949e">✅ 正确</div>
              </div>
              <div style="flex:1;min-width:65px;text-align:center;background:#161b22;border-radius:6px;padding:6px">
                <div style="font-size:1.3em;font-weight:700;color:#f85149">{wrong}</div>
                <div style="font-size:0.7em;color:#8b949e">❌ 错误</div>
              </div>
              <div style="flex:1;min-width:65px;text-align:center;background:#161b22;border-radius:6px;padding:6px">
                <div style="font-size:1.3em;font-weight:700;color:#d29922">{total_q}</div>
                <div style="font-size:0.7em;color:#8b949e">已批改</div>
              </div>
              <div style="flex:1;min-width:65px;text-align:center;background:#161b22;border-radius:6px;padding:6px">
                <div style="font-size:1.3em;font-weight:700;color:#c9d1d9">{len(exam_rounds_data)}</div>
                <div style="font-size:0.7em;color:#8b949e">轮次</div>
              </div>
            </div>'''

            # ── CSS 柱状图 + SVG 均分曲线 ──
            bar_w = max(3, min(16, int(600 / max(total_q, 1))))
            gap = max(1, min(3, bar_w // 4))
            chart_w = total_q * (bar_w + gap)
            chart_h = 140

            bars = ""
            for i, s in enumerate(q_scores):
                color = "#3fb950" if s >= 4 else "#d29922" if s == 3 else "#f85149"
                h_pct = s / 5 * 100
                x = i * (bar_w + gap)
                bars += f'<div style="position:absolute;bottom:0;left:{x}px;width:{bar_w}px;height:{h_pct}%;background:{color};border-radius:2px 2px 0 0;opacity:0.8" title="Q{i+1}: {s}/5"></div>'

            # SVG overlay for average line
            svg_w = chart_w
            avg_pts = " ".join(f"{i*(bar_w+gap)+bar_w/2:.1f},{chart_h - avgs2[i]/5*chart_h:.1f}" for i in range(total_q))

            # Round dividers
            round_lines = ""
            if has_rounds:
                acc = 0
                for rd in exam_rounds_data:
                    rt = rd.get("total", 0)
                    acc += rt
                    if acc < total_q:
                        x_div = acc * (bar_w + gap)
                        rn = rd.get("round", 0)
                        round_lines += f'<line x1="{x_div}" y1="0" x2="{x_div}" y2="{chart_h}" stroke="#8b949e" stroke-dasharray="3,3" opacity="0.5"/>'
                        round_lines += f'<text x="{x_div+3}" y="12" font-size="9" fill="#8b949e">R{rn+1}</text>'

            chart_html = f'''
            <div style="position:relative;height:{chart_h}px;overflow-x:auto;overflow-y:hidden;background:#161b22;border-radius:6px;padding:8px">
              <!-- 5分线 -->
              <div style="position:absolute;top:8px;left:0;right:0;border-top:1px dashed #21262d"></div>
              <div style="position:absolute;top:{int(chart_h*0.2)+8}px;left:0;right:0;border-top:1px dashed #21262d"></div>
              <div style="position:absolute;top:{int(chart_h*0.4)+8}px;left:0;right:0;border-top:1px dashed #21262d"></div>
              <div style="position:absolute;top:{int(chart_h*0.6)+8}px;left:0;right:0;border-top:1px dashed #21262d"></div>
              <div style="position:absolute;top:{int(chart_h*0.8)+8}px;left:0;right:0;border-top:1px dashed #21262d"></div>
              <!-- Bars -->
              <div style="position:relative;height:{chart_h}px;width:{chart_w}px;min-width:100%">
                {bars}
                <svg viewBox="0 0 {svg_w} {chart_h}" style="position:absolute;top:0;left:0;width:{chart_w}px;height:{chart_h}px;pointer-events:none">
                  <polyline points="{avg_pts}" fill="none" stroke="#58a6ff" stroke-width="2" opacity="0.9"/>
                  {round_lines}
                </svg>
              </div>
            </div>'''

            panels_html.append(f'''
            <div style="margin-bottom:8px">
              <div style="font-size:13px;font-weight:600;color:#c9d1d9;margin-bottom:6px">📊 做题成绩</div>
              {stat_cards}
              {chart_html}
              <div style="font-size:10px;color:#8b949e;margin-top:4px">
                <span style="color:#3fb950">■</span> ≥4分
                <span style="margin-left:8px;color:#d29922">■</span> 3分
                <span style="margin-left:8px;color:#f85149">■</span> ≤2分
                <span style="margin-left:8px;color:#58a6ff">━</span> 累计均分
              </div>
            </div>''')

        elif has_rounds:
            # 只有轮次汇总
            r_html = '<div style="display:flex;gap:6px;flex-wrap:wrap">'
            for rd in exam_rounds_data:
                avg = rd.get("avg", 0)
                c = rd.get("correct", 0)
                w = rd.get("wrong", 0)
                rn = rd.get("round", 0)
                bar_h = int(avg / 5 * 80)
                color = "#3fb950" if avg >= 3.5 else "#d29922" if avg >= 2.5 else "#f85149"
                r_html += f'''<div style="text-align:center;min-width:50px">
                  <div style="font-size:10px;color:#58a6ff;font-weight:700">{avg:.1f}</div>
                  <div style="height:80px;display:flex;align-items:flex-end;justify-content:center">
                    <div style="width:30px;height:{bar_h}px;background:{color};border-radius:3px 3px 0 0;opacity:0.8"></div>
                  </div>
                  <div style="font-size:10px;color:#8b949e">R{rn}</div>
                  <div style="font-size:9px;color:#3fb950">✓{c}</div>
                </div>'''
            r_html += '</div>'
            panels_html.append(f'''
            <div style="margin-bottom:8px">
              <div style="font-size:13px;font-weight:600;color:#c9d1d9;margin-bottom:6px">📊 轮次成绩</div>
              <div style="background:#161b22;border-radius:6px;padding:12px">{r_html}</div>
            </div>''')

    if not panels_html:
        return _empty_figure("等待数据中…")

    return f'<div style="font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:#0d1117;color:#c9d1d9;border-radius:8px;padding:12px">{"".join(panels_html)}</div>'


def _empty_figure(msg: str):
    """空白占位 HTML"""
    return f'''<div style="font-family:-apple-system,sans-serif;background:#0d1117;color:#8b949e;
        border-radius:8px;padding:40px;text-align:center;min-height:120px;
        display:flex;align-items:center;justify-content:center">
        <div><div style="font-size:1.8em;margin-bottom:6px">📊</div><div>{msg}</div></div></div>'''

def _format_task_table() -> List[List[str]]:
    rows: List[List[str]] = []
    for d in task_queue.get_all_tasks():
        tid = d.get("id", "")
        name = d.get("name", "")
        status = d.get("status", "")
        prog = d.get("progress", 0.0)
        info = (d.get("error") or d.get("message") or "")
        rows.append([tid, name, status, f"{float(prog):.0f}%", str(info)[:160]])
    return rows


# ---------------------------
# 断点续传 & 追加数据集
# ---------------------------

def _find_checkpoints(lora_name: str) -> List[str]:
    """扫描 LoRA 输出目录，找到所有 checkpoint-* 子目录"""
    out_dir = LORAS_DIR / lora_name
    if not out_dir.exists():
        return []
    checkpoints = []
    for d in sorted(out_dir.iterdir()):
        if d.is_dir() and d.name.startswith("checkpoint-"):
            # 验证有模型文件
            if (d / "trainer_state.json").exists() or (d / "optimizer.pt").exists() or (d / "model.safetensors").exists():
                checkpoints.append(d.name)
    return checkpoints


def _list_resumable_loras() -> List[str]:
    """列出所有有 checkpoint 的 LoRA（可断点续传）"""
    results = []
    if not LORAS_DIR.exists():
        return results
    for d in sorted(LORAS_DIR.iterdir()):
        if d.is_dir():
            ckpts = _find_checkpoints(d.name)
            if ckpts:
                results.append(f"{d.name} ({len(ckpts)} 个断点)")
    return results


def resume_training(
    lora_name: str,
    checkpoint_name: str,
    extra_datasets,     # 追加的数据集 (可选)
    extra_epochs: float,
):
    """从断点恢复训练，可选追加数据集"""
    if not lora_name:
        return "", "❌ 请选择要恢复的 LoRA"

    # 提取 lora_name（去掉 checkpoint 计数后缀）
    clean_name = lora_name.split(" (")[0].strip()
    out_dir = LORAS_DIR / clean_name

    if not out_dir.exists():
        return "", f"❌ LoRA 目录不存在: {out_dir}"

    # 读取训练元数据
    meta_path = out_dir / "forgex_meta.json"
    meta = {}
    if meta_path.exists():
        try:
            import json
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # 确定 checkpoint 路径
    if checkpoint_name and checkpoint_name != "最新":
        ckpt_path = str(out_dir / checkpoint_name)
    else:
        # 自动选最新的 checkpoint
        ckpts = _find_checkpoints(clean_name)
        if not ckpts:
            return "", f"❌ 没有找到断点: {out_dir}"
        ckpt_path = str(out_dir / ckpts[-1])

    if not Path(ckpt_path).exists():
        return "", f"❌ 断点目录不存在: {ckpt_path}"

    # 恢复训练参数
    base_model = meta.get("base_model", "")
    if not base_model:
        # 尝试从 adapter_config.json 读取
        adapter_cfg = out_dir / "adapter_config.json"
        if adapter_cfg.exists():
            try:
                import json
                acfg = json.loads(adapter_cfg.read_text(encoding="utf-8"))
                base_model = acfg.get("base_model_name_or_path", "")
            except Exception:
                pass
    if not base_model:
        return "", "❌ 无法确定基座模型。请在训练完成后保留 forgex_meta.json"

    # 构建数据集列表
    orig_datasets = meta.get("datasets", [])
    extra_ds = []
    if extra_datasets:
        if isinstance(extra_datasets, str):
            extra_ds = [extra_datasets] if extra_datasets else []
        else:
            extra_ds = [f for f in extra_datasets if f]

    all_datasets = orig_datasets + [str(DATASETS_DIR / f) for f in extra_ds if f not in [Path(d).name for d in orig_datasets]]

    if not all_datasets:
        return "", "❌ 没有数据集可用（原始元数据缺失且未选择追加数据集）"

    # 构建恢复参数
    orig_params = meta.get("params", {})
    params = {
        "output_name": clean_name,
        "lr": float(orig_params.get("lr", 2e-4)),
        "batch_size": int(orig_params.get("batch_size", 1)),
        "epochs": float(extra_epochs or orig_params.get("epochs", 1)),
        "max_seq_len": int(orig_params.get("max_seq_len", 4096)),
        "use_qlora": bool(orig_params.get("use_qlora", False)),
        "rank": int(orig_params.get("rank", 64)),
        "gradient_accumulation_steps": int(orig_params.get("gradient_accumulation_steps", 4)),
        "warmup_ratio": float(orig_params.get("warmup_ratio", 0.05)),
        "use_dora": bool(orig_params.get("use_dora", True)),
        "use_rslora": bool(orig_params.get("use_rslora", True)),
        "use_packing": bool(orig_params.get("use_packing", True)),
        "auto_clean": bool(orig_params.get("auto_clean", True)),
        "label_smoothing": float(orig_params.get("label_smoothing", 0.1)),
        "neftune_noise_alpha": float(orig_params.get("neftune_noise_alpha", 5.0)),
        "resume_from_checkpoint": ckpt_path,
    }

    method = meta.get("method", "SFT")
    extra_info = f" + 追加 {len(extra_ds)} 个数据集" if extra_ds else ""
    task_name = f"🔄 恢复: {clean_name}{extra_info}"

    def _run(task):
        result = trainer.train(
            method=method,
            backend="trl",
            base_model=base_model,
            dataset_path=all_datasets,
            params=params,
            task=task,
        )
        _invalidate_loras_cache()
        return result

    task_id = task_queue.submit(task_name, _run)
    ckpt_label = Path(ckpt_path).name
    msg = f"✅ 恢复训练已提交: {task_id}\n" \
          f"  断点: {ckpt_label}\n" \
          f"  基座: {Path(base_model).name}\n" \
          f"  数据集: {len(all_datasets)} 个"
    if extra_ds:
        msg += f"\n  追加: {', '.join(extra_ds)}"
    return task_id, msg

# ---------------------------
# Data Forge
# ---------------------------

def data_upload(file_objs, original_name: str):
    """上传一个或多个数据集文件"""
    if file_objs is None:
        return gr.update(), gr.update(), "未选择文件"
    # 兼容单文件和多文件
    if not isinstance(file_objs, (list, tuple)):
        file_objs = [file_objs]
    uploaded = []
    errors = []
    for file_obj in file_objs:
        src = Path(file_obj.name if hasattr(file_obj, 'name') else str(file_obj))
        try:
            # 多文件时不用 original_name（每个文件用自己的名字）
            oname = original_name if len(file_objs) == 1 else None
            meta = dm.upload(src, original_name=oname or src.name)
            uploaded.append(f"{meta['filename']} ({meta.get('stats', {}).get('count','?')} 条)")
        except Exception as e:
            errors.append(f"{src.name}: {e}")
    _invalidate_ds_cache()
    new_list = _list_datasets()
    last_name = uploaded[-1].split(" (")[0] if uploaded else (new_list[0] if new_list else None)
    up_single = gr.update(choices=new_list, value=last_name)   # 数据 Tab 单选
    up_multi = gr.update(choices=new_list)                      # 训练 Tab 多选
    msg_parts = []
    if uploaded:
        msg_parts.append(f"✅ 已上传 {len(uploaded)} 个: {', '.join(uploaded)}")
    if errors:
        msg_parts.append(f"❌ 失败 {len(errors)} 个: {'; '.join(errors)}")
    return up_single, up_multi, " | ".join(msg_parts) or "无操作"

def data_preview(filename: str, max_rows: int = 50):
    if not filename:
        return {"error":"请选择数据集"}
    try:
        rows = dm.preview(filename, n=int(max_rows))
        return {"count": len(rows), "samples": rows}
    except Exception as e:
        return {"error": f"预览失败: {e}"}

def data_clean_dedup(filename: str):
    if not filename:
        return "请先选择数据集"
    try:
        out = dm.deduplicate(filename)
        return f"✅ 去重完成: {out}"
    except Exception as e:
        return f"❌ 去重失敗: {e}"

def data_delete(filename: str):
    """删除数据集"""
    if not filename:
        return gr.update(), gr.update(), "请先选择要删除的数据集"
    try:
        dm.delete(filename)
        _invalidate_ds_cache()
        new_list = _list_datasets()
        up_single = gr.update(choices=new_list, value=new_list[0] if new_list else None)
        up_multi = gr.update(choices=new_list)  # multiselect 不预选
        return up_single, up_multi, f"✅ 已删除: {filename}"
    except Exception as e:
        return gr.update(), gr.update(), f"❌ 删除失败: {e}"

# ---------------------------
# Train Forge
# ---------------------------

def train_submit(
    base_model: str,
    dataset_files,  # str 或 list[str]
    method: str,
    output_name: str,
    lr: float,
    batch_size: int,
    epochs: float,
    max_seq_len: int,
    use_qlora: bool,
    recipe_name: str,
    rank: int = 64,
    ga_steps: int = 4,
    warmup_ratio: float = 0.05,
    auto_merge: bool = False,
    export_gguf: str = "",
    # v3.0 质量参数
    use_dora: bool = True,
    use_rslora: bool = True,
    use_packing: bool = True,
    auto_clean: bool = True,
    label_smoothing: float = 0.1,
    neftune_alpha: float = 5.0,
    # MoLoRA
    use_molora: bool = False,
    molora_n_experts: int = 4,
    molora_top_k: int = 2,
    molora_labels: str = "",
):
    if not base_model:
        return "", "❌ 请选择基座模型"
    # 兼容单选和多选
    if isinstance(dataset_files, str):
        dataset_files = [dataset_files] if dataset_files else []
    elif isinstance(dataset_files, (list, tuple)):
        dataset_files = [f for f in dataset_files if f]
    else:
        dataset_files = []
    if not dataset_files:
        return "", "❌ 请选择至少一个数据集"
    payload = dict(
        base_model=base_model,
        dataset=dataset_files,
        method=method,
        output_name=output_name,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        max_seq_len=max_seq_len,
        use_qlora=use_qlora,
        rank=int(rank),
        gradient_accumulation_steps=int(ga_steps),
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    saved = _save_recipe(recipe_name, payload) if recipe_name else ""
    ds_names = ", ".join(dataset_files)
    name = f"{method}: {output_name or 'run'}"
    # 构建数据集路径列表
    ds_path_list = [str(DATASETS_DIR / f) for f in dataset_files]
    def _run(task):
        params = {
            "output_name": output_name or "mylora",
            "lr": float(lr),
            "batch_size": int(batch_size),
            "epochs": float(epochs),
            "max_seq_len": int(max_seq_len),
            "use_qlora": bool(use_qlora),
            "rank": int(rank),
            "gradient_accumulation_steps": int(ga_steps),
            "warmup_ratio": float(warmup_ratio or 0.05),
            # v3.0 质量参数
            "use_dora": bool(use_dora),
            "use_rslora": bool(use_rslora),
            "use_packing": bool(use_packing),
            "auto_clean": bool(auto_clean),
            "label_smoothing": float(label_smoothing if label_smoothing else 0.1),
            "neftune_noise_alpha": float(neftune_alpha if neftune_alpha else 5.0),
            # MoLoRA
            "use_molora": bool(use_molora),
            "molora_n_experts": int(molora_n_experts or 4),
            "molora_top_k": int(molora_top_k or 2),
            "molora_expert_labels": [l.strip() for l in (molora_labels or "").split(",") if l.strip()] or None,
        }
        result = trainer.train(
            method=method,
            backend="trl",
            base_model=base_model,
            dataset_path=ds_path_list,  # 传入路径列表
            params=params,
            task=task,
        )
        _invalidate_loras_cache()  # 训练完成后刷新

        # ═══ 训练后自动合并 ═══
        _out_name = output_name or "mylora"
        lora_dir = LORAS_DIR / _out_name
        if auto_merge and lora_dir.exists():
            _safe_update_main(task, 96, "🔀 自动合并 LoRA → 完整模型...")
            try:
                merged_path = _post_train_merge(base_model, str(lora_dir), _out_name, task)
                if merged_path:
                    task.logs.append(f"✅ 合并完成: {merged_path}")
            except Exception as e:
                task.logs.append(f"⚠️ 合并失败（LoRA 仍可用）: {e}")

        # ═══ 训练后 GGUF 导出 ═══
        if export_gguf and lora_dir.exists():
            _safe_update_main(task, 98, f"📦 导出 GGUF ({export_gguf})...")
            try:
                _post_train_gguf(_out_name, export_gguf, task)
            except Exception as e:
                task.logs.append(f"⚠️ GGUF 导出失败: {e}")

        return result
    task_id = task_queue.submit(name, _run)
    # 构建特性列表
    _features = []
    if use_molora:
        _features.append(f"🧠MoLoRA({molora_n_experts}E×Top{molora_top_k})")
    else:
        if use_dora: _features.append("DoRA")
        if use_rslora: _features.append("rsLoRA")
    if use_packing: _features.append("Packing")
    if auto_clean: _features.append("AutoClean")
    if label_smoothing and label_smoothing > 0: _features.append(f"LS={label_smoothing}")
    _feat_str = " | ".join(_features) if _features else "标准"

    msg = f"✅ 任务已提交: {task_id} | 数据集: {ds_names} | 质量增强: {_feat_str}" + (f" | recipe={saved}" if saved else "")
    if auto_merge:
        msg += " | 🔀 训练后自动合并"
    if export_gguf:
        msg += f" | 📦 导出 {export_gguf}"
    return task_id, msg


# ════════════════════════════════════════════════════
#  从零定制 / 继续预训练
# ════════════════════════════════════════════════════

def pretrain_submit(
    preset: str,
    corpus_files,
    output_name: str,
    vocab_size: int = 32000,
    lr: float = 3e-4,
    batch_size: int = 1,
    epochs: float = 1,
    max_seq_len: int = 512,
    ga_steps: int = 8,
    max_steps: int = 0,
    # 自定义架构（preset == "custom" 时使用）
    custom_hidden: int = 768,
    custom_layers: int = 12,
    custom_heads: int = 12,
    custom_kv_heads: int = 4,
    custom_intermediate: int = 2048,
    custom_max_pos: int = 2048,
):
    """提交从零预训练任务"""
    if isinstance(corpus_files, str):
        corpus_files = [corpus_files] if corpus_files else []
    elif isinstance(corpus_files, (list, tuple)):
        corpus_files = [f for f in corpus_files if f]
    else:
        corpus_files = []
    if not corpus_files:
        return "", "❌ 请选择训练语料"
    if not output_name:
        output_name = f"pretrain_{preset}_{int(time.time()) % 10000}"

    from core.pretrain import ArchConfig, ARCH_PRESETS, pretrain_engine, train_tokenizer

    # 构建架构
    if preset == "custom":
        arch = ArchConfig(
            preset="custom",
            hidden_size=int(custom_hidden),
            num_hidden_layers=int(custom_layers),
            num_attention_heads=int(custom_heads),
            num_key_value_heads=int(custom_kv_heads),
            intermediate_size=int(custom_intermediate),
            max_position_embeddings=int(custom_max_pos),
            vocab_size=int(vocab_size),
        )
    else:
        arch = ArchConfig.from_preset(preset, vocab_size=int(vocab_size))

    param_str = arch.param_count_str()
    vram_est = arch.estimate_vram_mb(batch_size=int(batch_size), seq_len=int(max_seq_len))

    corpus_path_list = [str(DATASETS_DIR / f) for f in corpus_files]
    name = f"预训练: {output_name} ({param_str})"

    def _run(task):
        # 1. 训练 tokenizer
        tok_dir = train_tokenizer(
            [Path(p) for p in corpus_path_list],
            vocab_size=int(vocab_size),
            output_dir=LORAS_DIR / output_name / "_tokenizer",
            task=task,
        )
        # 2. 预训练
        return pretrain_engine.pretrain(
            arch_config=arch,
            corpus_paths=corpus_path_list,
            tokenizer_path=tok_dir,
            output_name=output_name,
            params={
                "lr": float(lr), "batch_size": int(batch_size),
                "epochs": float(epochs), "max_seq_len": int(max_seq_len),
                "gradient_accumulation_steps": int(ga_steps),
                "max_steps": int(max_steps) if max_steps else None,
            },
            task=task,
        )

    task_id = task_queue.submit(name, _run)
    offload = " ⚡CPU offload" if arch.cpu_offload else ""
    return task_id, (
        f"✅ 从零预训练已提交: {task_id}\n"
        f"   架构: {preset} ({param_str}) | 预估 VRAM: ~{vram_est}MB{offload}\n"
        f"   语料: {', '.join(corpus_files)} | vocab: {vocab_size}"
    )


def cpt_submit(
    base_model_mode: str,
    base_model_hf: str,
    base_model_local: str,
    corpus_files,
    output_name: str,
    lr: float = 1e-4,
    batch_size: int = 1,
    epochs: float = 1,
    max_seq_len: int = 512,
    ga_steps: int = 8,
    max_steps: int = 0,
):
    """提交继续预训练 (CPT) 任务"""
    # 解析基座模型
    base_model = ""
    if base_model_mode == "HuggingFace 模型 ID":
        base_model = (base_model_hf or "").strip()
    else:
        raw_label = (base_model_local or "").strip()
        if raw_label:
            clean_name = _extract_model_name_from_label(raw_label)
            base_model = str(LORAS_DIR / clean_name)
    if not base_model:
        return "", "❌ 请选择基座模型"

    if isinstance(corpus_files, str):
        corpus_files = [corpus_files] if corpus_files else []
    elif isinstance(corpus_files, (list, tuple)):
        corpus_files = [f for f in corpus_files if f]
    else:
        corpus_files = []
    if not corpus_files:
        return "", "❌ 请选择训练语料"
    if not output_name:
        output_name = f"cpt_{int(time.time()) % 10000}"

    corpus_path_list = [str(DATASETS_DIR / f) for f in corpus_files]
    name = f"继续预训练: {output_name}"

    from core.pretrain import pretrain_engine

    def _run(task):
        return pretrain_engine.continual_pretrain(
            base_model=base_model,
            corpus_paths=corpus_path_list,
            output_name=output_name,
            params={
                "lr": float(lr), "batch_size": int(batch_size),
                "epochs": float(epochs), "max_seq_len": int(max_seq_len),
                "gradient_accumulation_steps": int(ga_steps),
                "max_steps": int(max_steps) if max_steps else None,
            },
            task=task,
        )

    task_id = task_queue.submit(name, _run)
    return task_id, (
        f"✅ 继续预训练已提交: {task_id}\n"
        f"   基座: {base_model} | 语料: {', '.join(corpus_files)}\n"
        f"   全参数训练（非 LoRA）| lr={lr}"
    )


def knowledge_inject_submit(
    model_mode: str, model_hf: str, model_local: str,
    doc_files, api_base: str, api_key: str, api_model: str,
    output_name: str, qa_per_chunk: int = 3, max_chunks: int = 200,
    use_cot: bool = True, system_prompt: str = "",
    lr: float = 2e-4, batch_size: int = 2, epochs: float = 3,
    max_seq: int = 2048, rank: int = 64, use_qlora: bool = False,
    data_only: bool = False,
):
    """提交知识注入任务"""
    # 解析模型
    base_model = ""
    if model_mode == "HuggingFace 模型 ID":
        base_model = (model_hf or "").strip()
    else:
        raw_label = (model_local or "").strip()
        if raw_label:
            clean_name = _extract_model_name_from_label(raw_label)
            base_model = str(LORAS_DIR / clean_name)
    if not base_model and not data_only:
        return "", "❌ 请选择目标模型"

    if isinstance(doc_files, str):
        doc_files = [doc_files] if doc_files else []
    elif isinstance(doc_files, (list, tuple)):
        doc_files = [f for f in doc_files if f]
    else:
        doc_files = []
    if not doc_files:
        return "", "❌ 请选择知识文档"

    if not api_base or not api_model:
        return "", "❌ 请配置 API（Base URL + 模型名）"

    if not output_name:
        output_name = f"knowledge_{int(time.time()) % 10000}"

    doc_path_list = [str(DATASETS_DIR / f) for f in doc_files]

    from core.pretrain import knowledge_forge

    if data_only:
        name = f"知识提取: {output_name}"
        def _run(task):
            return knowledge_forge.generate_data_only(
                doc_paths=doc_path_list,
                api_base=api_base, api_key=api_key, api_model=api_model,
                output_name=output_name,
                qa_per_chunk=int(qa_per_chunk),
                max_chunks=int(max_chunks),
                system_prompt=system_prompt,
                use_cot=use_cot,
                task=task,
            )
    else:
        name = f"知识注入: {output_name}"
        def _run(task):
            return knowledge_forge.run(
                model_path=base_model,
                doc_paths=doc_path_list,
                api_base=api_base, api_key=api_key, api_model=api_model,
                output_name=output_name,
                qa_per_chunk=int(qa_per_chunk),
                max_chunks=int(max_chunks),
                use_cot=use_cot,
                system_prompt=system_prompt,
                train_params={
                    "lr": float(lr), "batch_size": int(batch_size),
                    "epochs": float(epochs), "max_seq_len": int(max_seq),
                    "rank": int(rank), "use_qlora": use_qlora,
                },
                task=task,
            )

    task_id = task_queue.submit(name, _run)
    _mode = "仅生成数据" if data_only else "知识注入 (API→QA→SFT)"
    return task_id, (
        f"✅ {_mode}已提交: {task_id}\n"
        f"   文档: {', '.join(doc_files)} | API: {api_model}\n"
        f"   QA/块={qa_per_chunk} | CoT={'是' if use_cot else '否'}"
        + ("" if data_only else f" | 目标: {base_model}")
    )


def _safe_update_main(task, pct, msg):
    if task:
        try: task.update_progress(pct, msg)
        except Exception: pass


def _post_train_merge(base_model: str, lora_dir: str, output_name: str, task=None) -> str:
    """训练后自动合并 LoRA 到基座模型"""
    from pathlib import Path
    import json as _json

    lora_p = Path(lora_dir)
    adapter_cfg = lora_p / "adapter_config.json"

    # 确定基座模型
    if adapter_cfg.exists():
        try:
            acfg = _json.loads(adapter_cfg.read_text(encoding="utf-8"))
            real_base = acfg.get("base_model_name_or_path", "")
            if real_base:
                base_model = real_base
        except Exception:
            pass

    if not base_model:
        return ""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    log(f"合并 LoRA: {lora_dir} + {base_model}")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    load_kw = dict(torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True)
    if torch.cuda.is_available():
        load_kw["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kw)
    try:
        tokenizer = AutoTokenizer.from_pretrained(lora_dir, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    model = PeftModel.from_pretrained(model, lora_dir)
    model = model.merge_and_unload()

    merged_dir = LORAS_DIR / f"{output_name}_merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))

    del model
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log(f"✅ 合并完成: {merged_dir}")
    _invalidate_loras_cache()
    return str(merged_dir)


def _post_train_gguf(output_name: str, quant_type: str, task=None):
    """训练后自动导出 GGUF"""
    # 优先用合并模型，没有则先合并
    merged_dir = LORAS_DIR / f"{output_name}_merged"
    lora_dir = LORAS_DIR / output_name

    source = str(merged_dir) if merged_dir.exists() else str(lora_dir)
    try:
        from core.exporter import export_to_gguf
        export_to_gguf(source, quant_type=quant_type, task=task)
    except ImportError:
        if task:
            task.logs.append("⚠️ GGUF 导出模块不可用，请安装 llama-cpp-python")
    except Exception as e:
        if task:
            task.logs.append(f"⚠️ GGUF 导出失败: {e}")

def _resolve_base_model_for_training(model_mode: str, hf_id: str, local_model_path: str) -> str:
    mode = (model_mode or "").strip()
    if mode.startswith("HF"):
        return (hf_id or "").strip()
    if "本地" in mode:
        path_str = (local_model_path or "").strip()
        if not path_str:
            return ""
        p = Path(path_str)
        # 如果是 LoRA 适配器目录 → 自动读取基座模型
        if p.is_dir() and (p / "adapter_config.json").exists() and not (p / "config.json").exists():
            try:
                import json
                acfg = json.loads((p / "adapter_config.json").read_text(encoding="utf-8"))
                base = acfg.get("base_model_name_or_path", "")
                if base:
                    from core import log
                    log(f"自动检测 LoRA 适配器，基座: {base}，LoRA: {path_str}")
                    return base  # 返回基座模型路径
            except Exception:
                pass
        return path_str
    # GGUF is not trainable in this pipeline
    return ""

def _resolve_lora_for_continued_training(local_model_path: str) -> str:
    """如果路径是 LoRA 适配器目录，返回 LoRA 路径（用于继续训练时叠加）"""
    if not local_model_path:
        return ""
    p = Path(local_model_path.strip())
    if p.is_dir() and (p / "adapter_config.json").exists() and not (p / "config.json").exists():
        return str(p)
    return ""

def train_submit_ui(
    model_mode: str,
    hf_id: str,
    local_model_path: str,
    dataset_files,  # str 或 list[str]（多选）
    method: str,
    output_name: str,
    lr: float,
    batch_size: int,
    epochs: float,
    max_seq_len: int,
    use_qlora: bool,
    recipe_name: str,
    rank: int = 64,
    ga_steps: int = 4,
    warmup_ratio: float = 0.05,
    auto_merge: bool = False,
    export_gguf: str = "",
    # v3.0 质量参数
    use_dora: bool = True,
    use_rslora: bool = True,
    use_packing: bool = True,
    auto_clean: bool = True,
    label_smoothing: float = 0.1,
    neftune_alpha: float = 5.0,
    # MoLoRA
    use_molora: bool = False,
    molora_n_experts: int = 4,
    molora_top_k: int = 2,
    molora_labels: str = "",
):
    base_model = _resolve_base_model_for_training(model_mode, hf_id, local_model_path)
    if not base_model:
        return "", "❌ 训练只支持『HF 模型ID』或『本地模型路径』。GGUF 格式无法进行微调训练，仅可用于聊天测试和 Ollama 部署。"

    # 检测是否选了 LoRA 目录 → 提示用户（继续训练 = 在同一基座上再训一个新 LoRA）
    lora_hint = _resolve_lora_for_continued_training(local_model_path)
    extra_msg = ""
    if lora_hint:
        lora_name = Path(lora_hint).name
        extra_msg = f" | 检测到 LoRA [{lora_name}]，将在同一基座上训练新 LoRA（原 LoRA 不影响）"
    tid, msg = train_submit(
        base_model,
        dataset_files,
        method,
        output_name,
        lr,
        batch_size,
        epochs,
        max_seq_len,
        use_qlora,
        recipe_name,
        rank,
        ga_steps,
        warmup_ratio,
        auto_merge=auto_merge,
        export_gguf=export_gguf or "",
        use_dora=use_dora,
        use_rslora=use_rslora,
        use_packing=use_packing,
        auto_clean=auto_clean,
        label_smoothing=label_smoothing,
        neftune_alpha=neftune_alpha,
        use_molora=use_molora,
        molora_n_experts=int(molora_n_experts or 4),
        molora_top_k=int(molora_top_k or 2),
        molora_labels=str(molora_labels or ""),
    )
    return tid, msg + extra_msg

def train_methods():
    """返回可用训练方法列表。首次用硬编码避免启动时 import TRL（很慢）"""
    try:
        methods = trainer.available_methods()
        return [m.method.upper() for m in methods if m.available]
    except Exception:
        return ["SFT"]

# 启动时用硬编码，不触发 TRL import（节省 3-10 秒）
_STARTUP_METHODS = ["SFT", "DPO", "ORPO", "KTO"]

def train_load_recipe(recipe_file: str):
    r = _load_recipe(recipe_file)
    if not r:
        return [gr.update()] * 10
    # dataset 可能是字符串（旧配方）或列表（新配方）
    ds_val = r.get("dataset", "")
    if isinstance(ds_val, str):
        ds_val = [ds_val] if ds_val else []
    return [
        gr.update(value=r.get("base_model","")),
        gr.update(value=ds_val),
        gr.update(value=r.get("method","SFT")),
        gr.update(value=r.get("output_name","")),
        gr.update(value=float(r.get("lr", 2e-4))),
        gr.update(value=int(r.get("batch_size", 1))),
        gr.update(value=float(r.get("epochs", 1.0))),
        gr.update(value=int(r.get("max_seq_len", 2048))),
        gr.update(value=bool(r.get("use_qlora", False))),
        gr.update(value=r.get("recipe_name","")),
    ]

# ---------------------------
# 知识膨胀 (Expansion)
# ---------------------------

def expansion_depth_submit(
    model_mode, hf_id, local_model, output, strategy, num_layers, noise
):
    """深度膨胀提交"""
    base = _resolve_model_input(model_mode, hf_id, local_model)
    if not base:
        return "", "❌ 请选择源模型"
    from core.expansion import depth_expand
    def _run(task):
        return depth_expand(
            base, output or "depth_expanded",
            strategy=strategy, num_new_layers=int(num_layers or 8),
            noise_scale=float(noise or 0.01), task=task,
        )
    tid = task_queue.submit(f"深度膨胀: {output}", _run)
    return tid, f"✅ 深度膨胀已提交: {tid}"

def expansion_width_submit(
    model_mode, hf_id, local_model, output, target_h, target_inter, noise
):
    """宽度膨胀提交"""
    base = _resolve_model_input(model_mode, hf_id, local_model)
    if not base:
        return "", "❌ 请选择源模型"
    from core.expansion import width_expand
    def _run(task):
        return width_expand(
            base, output or "width_expanded",
            target_hidden=int(target_h or 0),
            target_intermediate=int(target_inter or 0),
            noise_scale=float(noise or 0.01), task=task,
        )
    tid = task_queue.submit(f"宽度膨胀: {output}", _run)
    return tid, f"✅ 宽度膨胀已提交: {tid}"

def expansion_graft_submit(
    small_mode, small_hf, small_local,
    large_mode, large_hf, large_local,
    output, graft_pos, num_graft, noise,
):
    """知识嫁接提交"""
    small = _resolve_model_input(small_mode, small_hf, small_local)
    large = _resolve_model_input(large_mode, large_hf, large_local)
    if not small:
        return "", "❌ 请选择小模型"
    if not large:
        return "", "❌ 请选择大模型（知识源）"
    from core.expansion import knowledge_graft
    def _run(task):
        return knowledge_graft(
            small, large, output or "grafted_model",
            graft_layers=graft_pos, num_graft_layers=int(num_graft or 4),
            noise_scale=float(noise or 0.005), task=task,
        )
    tid = task_queue.submit(f"知识嫁接: {output}", _run)
    return tid, f"✅ 知识嫁接已提交: {tid}"

def expansion_vocab_submit(
    model_mode, hf_id, local_model, output, new_tokens_text, target_vocab, init_strat
):
    """词表扩展提交"""
    base = _resolve_model_input(model_mode, hf_id, local_model)
    if not base:
        return "", "❌ 请选择源模型"
    from core.expansion import vocab_expand
    tokens = [t.strip() for t in (new_tokens_text or "").split(",") if t.strip()]
    def _run(task):
        return vocab_expand(
            base, output or "vocab_expanded",
            new_tokens=tokens if tokens else None,
            new_vocab_size=int(target_vocab or 0),
            init_strategy=init_strat, task=task,
        )
    tid = task_queue.submit(f"词表扩展: {output}", _run)
    return tid, f"✅ 词表扩展已提交: {tid}"

def expansion_hybrid_submit(
    model_mode, hf_id, local_model, output,
    target_layers, target_hidden, target_inter, depth_strat, noise,
):
    """混合膨胀提交"""
    base = _resolve_model_input(model_mode, hf_id, local_model)
    if not base:
        return "", "❌ 请选择源模型"
    from core.expansion import hybrid_expand
    def _run(task):
        return hybrid_expand(
            base, output or "hybrid_expanded",
            target_layers=int(target_layers or 0),
            target_hidden=int(target_hidden or 0),
            target_intermediate=int(target_inter or 0),
            depth_strategy=depth_strat,
            noise_scale=float(noise or 0.01), task=task,
        )
    tid = task_queue.submit(f"混合膨胀: {output}", _run)
    return tid, f"✅ 混合膨胀已提交: {tid}"

def expansion_moe_submit(
    model_mode, hf_id, local_model, output,
    num_experts, top_k, moe_layers_str, noise,
):
    """MoE 专家嫁接提交"""
    base = _resolve_model_input(model_mode, hf_id, local_model)
    if not base:
        return "", "❌ 请选择源模型"
    from core.expansion import moe_upcycle
    def _run(task):
        return moe_upcycle(
            base, output or "moe_model",
            num_experts=int(num_experts or 4),
            top_k=int(top_k or 2),
            moe_layers=moe_layers_str or "all",
            noise_scale=float(noise or 0.01),
            task=task,
        )
    tid = task_queue.submit(f"MoE嫁接: {output}", _run)
    return tid, f"✅ MoE 专家嫁接已提交: {tid}"

def _resolve_model_input(model_mode, hf_id, local_model):
    """通用: 从 UI 输入解析模型路径"""
    if model_mode == "HuggingFace 模型 ID":
        return (hf_id or "").strip()
    else:
        raw = (local_model or "").strip()
        if raw:
            return _resolve_model_path(raw)
    return ""

def native_hybrid_submit(base, method, models, weight, density, out, franken_text):
    """原生杂交融合提交"""
    if not base:
        return "", "❌ 请填写基座模型"
    if not models or len(models) < 1:
        return "", "❌ 请选择至少 1 个模型"
    if method == "slerp" and len(models) != 1:
        return "", "❌ SLERP 需要恰好 1 个模型（与基座插值）"

    model_paths = [str(LORAS_DIR / m) for m in models]

    # 解析 franken spec
    franken_spec = None
    if method == "frankenmerge" and franken_text:
        try:
            import json
            franken_spec = json.loads(franken_text)
        except Exception:
            return "", "❌ Frankenmerge 规格必须是有效 JSON"

    def _run(task):
        result = native_merger.merge(
            base_model=base, model_paths=model_paths,
            method=method, output_name=out or "hybrid_model",
            t=float(weight), density=float(density),
            franken_spec=franken_spec, task=task,
        )
        _invalidate_loras_cache()
        return result
    tid = task_queue.submit(f"NativeMerge({method}): {out}", _run)
    return tid, f"✅ 原生融合已提交: {tid}"


# ---------------------------
# Merge & Export Forge
# ---------------------------

def merge_submit(base_model: str, adapters: List[str], output_name: str):
    if not base_model or not adapters:
        return "", "❌ 请选择基座模型和至少一个 LoRA 适配器"
    adapter_paths = [str(LORAS_DIR / a) for a in adapters]
    def _run(task):
        return merger.merge_multiple_adapters_to_base(
            base_model=base_model,
            adapter_paths=adapter_paths,
            output_name=output_name or "merged_model",
            task=task,
        )
    task_id = task_queue.submit(f"Merge: {output_name or 'merged_model'}", _run)
    return task_id, f"✅ Merge 任务已提交: {task_id}"

def export_gguf_submit(model_dir: str, out_name: str, quant: str):
    if not model_dir:
        return "", "❌ 请选择模型目录"
    def _run(task):
        return exporter.export_gguf(model_dir=model_dir, output_name=out_name, quant=quant, task=task)
    task_id = task_queue.submit(f"GGUF: {out_name or 'gguf'}", _run)
    return task_id, f"✅ GGUF 任务已提交: {task_id}"

def export_ollama_submit(gguf_path: str, model_name: str):
    if not gguf_path or not model_name:
        return "", "❌ 请选择 GGUF 路径和模型名"
    def _run(task):
        return exporter.export_ollama(gguf_path=gguf_path, model_name=model_name, task=task)
    task_id = task_queue.submit(f"Ollama: {model_name}", _run)
    return task_id, f"✅ Ollama 任务已提交: {task_id}"

# ---------------------------
# Bench Forge
# ---------------------------

def bench_submit(model_dir: str, quick_set: str):
    if not model_dir:
        return "", "❌ 请选择模型"
    def _run(task):
        return bench.quick_test(model_path=model_dir, task=task)
    task_id = task_queue.submit(f"Bench: {quick_set}", _run)
    return task_id, f"✅ 评测任务已提交: {task_id}"

# ---------------------------
# Monitor
# ---------------------------

def monitor_refresh_v2(task_id: str):
    """增强版监控刷新：状态/进度/日志/多功能图表/任务表。"""
    try:
        data = _poll_task(task_id) if task_id else {"status": "", "progress": 0, "message": "", "logs": []}
        table = _format_task_table()
        log_lines = data.get("logs") or []
        # 过滤掉 [METRIC] 行（只供图表用，不显示给用户）
        display_lines = [str(x) for x in log_lines if not str(x).startswith("[METRIC]")]
        logs_text = "\n".join(display_lines[-200:])
        progress = float(data.get("progress", 0) or 0)
        status_text = f"状态: {data.get('status','')} | {data.get('message','')}"
        if data.get("error"):
            status_text += f"\n❌ 错误: {data['error']}"
        fig = _make_loss_figure(log_lines)
        return status_text, progress, logs_text, fig, table
    except Exception as e:
        err_fig = _empty_figure(f"刷新异常：{e}")
        return f"❌ 刷新失败：{e}", 0.0, "", err_fig, _format_task_table()

def cancel_task(task_id: str):
    """取消任务"""
    if not task_id:
        return "请输入要取消的任务ID"
    try:
        success = task_queue.cancel_task(task_id)
        if success:
            return f"✅ 任务已取消: {task_id}"
        else:
            return f"⚠️ 无法取消任务（可能已完成或不存在）: {task_id}"
    except Exception as e:
        return f"❌ 取消失败: {e}"

_env_report_cache = None
def _env_report_md() -> str:
    """环境自检（结果缓存，只慢一次）"""
    global _env_report_cache
    if _env_report_cache is not None:
        return _env_report_cache
    try:
        import platform, sys
        import gradio as gr
        items = []
        items.append(f"- Python: {sys.version.split()[0]} ({platform.system()} {platform.release()})")
        items.append(f"- Gradio: {getattr(gr, '__version__', 'unknown')}")
        try:
            import torch
            items.append(f"- Torch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | Device count: {torch.cuda.device_count()}")
            if torch.cuda.is_available():
                items.append(f"  - GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_mem // 1024**3}GB")
                items.append(f"  - BF16: {torch.cuda.is_bf16_supported()} | 推荐精度: {'bf16' if torch.cuda.is_bf16_supported() else 'fp16'}")
        except Exception as e:
            items.append(f"- Torch: unavailable ({e})")
        for pkg in ["transformers","peft","accelerate","datasets","trl","bitsandbytes"]:
            try:
                m=__import__(pkg)
                items.append(f"- {pkg}: {getattr(m,'__version__','unknown')}")
            except Exception as e:
                items.append(f"- {pkg}: ❌ not installed (`pip install {pkg}`)")
        try:
            import flash_attn
            items.append(f"- flash_attn: {getattr(flash_attn, '__version__', 'installed')} ✅")
        except ImportError:
            items.append("- flash_attn: not installed（可选加速，`pip install flash-attn --no-build-isolation`）")
        _env_report_cache = "### 环境信息\n" + "\n".join(items)
        return _env_report_cache
    except Exception as e:
        return f"环境自检失败：{e}"

def _env_report_quick() -> str:
    """快速环境摘要（用缓存的 GPU 信息，不额外 import）"""
    import platform, sys
    lines = [f"- Python: {sys.version.split()[0]} ({platform.system()})"]
    gpu = _get_gpu_cached()
    if gpu and gpu.get("vram_mb", 0) > 0:
        lines.append(f"- GPU: {gpu.get('name','?')} | VRAM: {gpu['vram_mb']//1024}GB")
    else:
        lines.append("- GPU: N/A (CPU mode)")
    lines.append("- 点击「刷新环境详情」查看完整依赖版本")
    return "### 环境信息（快速）\n" + "\n".join(lines)

def _make_chat_fn(api_type: str, base_url: str, api_key: str,
                  model_name: str):
    """创建推理函数，用于合成数据生成和一键锻造。"""
    if api_type == "openai":
        if not api_key:
            return None

        is_anthropic = ("anthropic" in (base_url or "").lower()
                        or (model_name or "").lower().startswith("claude"))

        def _chat(prompt: str) -> str:
            import urllib.request, urllib.error, json as _j, ssl

            if is_anthropic:
                base = (base_url or "https://api.anthropic.com").rstrip("/")
                if not base.endswith("/v1"):
                    base += "/v1"
                url = f"{base}/messages"
                payload = _j.dumps({
                    "model": model_name or "claude-sonnet-4-20250514",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1024, "temperature": 0.8,
                }).encode()
                headers = {
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                }
            else:
                url = (base_url.rstrip("/") + "/chat/completions")
                payload = _j.dumps({
                    "model": model_name or "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1024, "temperature": 0.8,
                }).encode()
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                }

            req = urllib.request.Request(url, data=payload, headers=headers)

            # 先尝试正常 SSL，失败再降级
            data = None
            for ssl_mode in ["normal", "no_verify"]:
                try:
                    if ssl_mode == "normal":
                        ctx = ssl.create_default_context()
                    else:
                        ctx = ssl.create_default_context()
                        ctx.check_hostname = False
                        ctx.verify_mode = ssl.CERT_NONE
                    req = urllib.request.Request(url, data=payload, headers=headers)
                    with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
                        data = _j.loads(resp.read())
                    break
                except (ssl.SSLError, urllib.error.URLError) as e:
                    if ssl_mode == "no_verify":
                        raise RuntimeError(f"API 连接失败: {e}") from e
                    continue
                except urllib.error.HTTPError as e:
                    err_body = ""
                    try: err_body = e.read().decode("utf-8", errors="replace")[:300]
                    except Exception: pass
                    raise RuntimeError(f"API 请求失败 ({e.code}): {err_body}") from e

            if data is None:
                raise RuntimeError("API 请求失败: 无响应数据")

            if "error" in data:
                err = data["error"]
                msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                raise RuntimeError(f"API 错误: {msg}")

            # 解析响应（兼容 OpenAI 和 Anthropic 格式）
            if is_anthropic:
                blocks = data.get("content", [])
                return "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            else:
                choices = data.get("choices", [])
                if not choices:
                    raise RuntimeError(f"API 返回空结果")
                return choices[0].get("message", {}).get("content", "")
        return _chat
    else:
        # local — placeholder, 未来接 chat_engine
        return None


def _apply_train_preset(name: str):
    # 返回 lr, batch, epochs, max_seq, qlora, rank, ga (gradient_accumulation_steps)
    name = (name or "").strip()
    if "快速冒烟" in name:
        return 5e-4, 1, 0.2, 1024, False, 16, 1
    if "低显存" in name:
        return 1e-4, 1, 1.0, 1024, True, 32, 4
    if "自动" in name:
        # 根据 GPU VRAM 自动推荐最优参数
        gpu = _get_gpu_cached()
        vram = gpu.get("vram_mb", 0) if gpu else 0
        if vram >= 24000:      # 24GB+ (3090/4090/A100)
            return 2e-4, 2, 1.0, 4096, False, 64, 4
        elif vram >= 12000:    # 12-24GB (3060 12G/4070+)
            return 2e-4, 1, 1.0, 2048, False, 64, 4
        elif vram >= 8000:     # 8-12GB (3060 8G/4060)
            return 1e-4, 1, 1.0, 1024, True, 32, 4
        else:                  # <8GB
            return 1e-4, 1, 1.0, 512, True, 16, 8
    # 稳定高质量
    return 2e-4, 1, 1.0, 2048, False, 64, 4


def _list_training_history() -> List[List[str]]:
    """扫描 data/loras/*/forgex_meta.json 或 forgex_pretrain_meta.json，列出训练历史"""
    rows = []
    for d in sorted(LORAS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        # 检查所有 meta 文件类型
        meta_path = d / "forgex_meta.json"
        pretrain_meta = d / "forgex_pretrain_meta.json"
        mm_meta = d / "forgex_multimodal_meta.json"
        if mm_meta.exists():
            meta_path = mm_meta
        elif pretrain_meta.exists():
            meta_path = pretrain_meta
        if not meta_path.exists():
            # 尝试读取 adapter_config.json
            ac = d / "adapter_config.json"
            if ac.exists():
                rows.append([d.name, "?", "?", "?", "?", d.stat().st_mtime])
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))

            _type = meta.get("type", "")
            mode = meta.get("mode", "sft")

            if _type == "multimodal":
                # 多模态模型
                modalities = []
                if meta.get("enable_vision"):
                    modalities.append(f"👁️{meta.get('vision_encoder', '?')}")
                if meta.get("enable_audio"):
                    modalities.append(f"🔊{meta.get('audio_encoder', '?')}")
                _base = meta.get("base_model", "?").split("/")[-1]
                _params_str = f"🔮多模态 r={meta.get('lora_rank', '?')} {' '.join(modalities)}"

                rows.append([
                    d.name, _base, " + ".join(modalities), _params_str,
                    meta.get("timestamp", "?"), meta.get("timestamp", ""),
                ])

            elif mode in ("pretrain", "cpt"):
                # 预训练模型
                _mode_label = "🔨从零" if mode == "pretrain" else "📚CPT"
                _base = meta.get("base_model", "")
                if mode == "pretrain":
                    arch = meta.get("architecture", {})
                    _base = f"{arch.get('preset', '?')} ({meta.get('total_params', 0) / 1e6:.0f}M)"
                else:
                    _base = _base.split("/")[-1] if _base else "?"
                _loss_str = f"loss={meta['final_loss']:.3f}" if meta.get("final_loss") is not None else ""
                _tokens = meta.get("total_tokens", 0)
                _data_str = f"{_tokens / 1e6:.1f}M tokens" if _tokens else "?"
                _params_str = f"{_mode_label} lr={meta.get('lr', '?')}"
                if _loss_str:
                    _params_str += f" {_loss_str}"

                rows.append([
                    d.name, _base, _data_str, _params_str,
                    meta.get("timestamp", "?"), meta.get("timestamp", ""),
                ])
            else:
                # 标准 SFT/DPO 训练
                _quality_tags = []
                if meta.get("dora"): _quality_tags.append("DoRA")
                if meta.get("rslora"): _quality_tags.append("rsLoRA")
                if meta.get("sample_packing"): _quality_tags.append("Pack")
                if meta.get("label_smoothing", 0) > 0: _quality_tags.append(f"LS")
                _loss_str = ""
                if meta.get("final_loss") is not None:
                    _loss_str = f"loss={meta['final_loss']:.3f}"
                if meta.get("eval_loss") is not None:
                    _loss_str += f" eval={meta['eval_loss']:.3f}"
                _quality_str = " ".join(_quality_tags) if _quality_tags else ""
                _params_str = f"r={meta.get('rank','?')} lr={meta.get('lr','?')}"
                if _quality_str:
                    _params_str += f" [{_quality_str}]"
                if _loss_str:
                    _params_str += f" {_loss_str}"

                rows.append([
                    d.name,
                    meta.get("base_model", "?").split("/")[-1],
                    str(meta.get("dataset_size", "?")),
                    _params_str,
                    meta.get("timestamp", "?"),
                    meta.get("timestamp", ""),
                ])
        except Exception:
            rows.append([d.name, "?", "?", "?", "?", ""])
    return rows


def _vram_estimate_detailed(model_id: str, qlora_on: bool, rank_val: int, seq_len: int, batch_size: int = 1):
    """精细 VRAM 估算 + 训练时间预测"""
    from core.utils import estimate_vram_mb, guess_model_size
    model_id = str(model_id or "")
    if not model_id:
        return "💡 请先选择模型"
    size_b = guess_model_size(model_id)
    method = "qlora" if qlora_on else "lora"
    est = estimate_vram_mb(size_b, method, int(batch_size or 1), int(seq_len or 2048), int(rank_val or 64), True)

    gpu = _get_gpu_cached()
    vram_gb = gpu.get("vram_mb", 0) / 1024 if gpu else 0

    bar = ""
    if vram_gb > 0:
        pct = min(est["total_gb"] / vram_gb * 100, 150)
        filled = int(pct / 5)
        bar = "█" * min(filled, 20) + "░" * max(0, 20 - filled)
        bar = f"[{bar}] {pct:.0f}%"
        if pct > 95:
            status = "🔴 可能 OOM"
        elif pct > 80:
            status = "🟡 偏紧"
        else:
            status = "🟢 OK"
    else:
        status = "❓ 无 GPU"
        bar = ""

    lines = [
        f"📊 {size_b}B 模型 | {method.upper()} | r={rank_val} | seq={seq_len}",
        f"   模型权重: {est['model_mb']/1024:.1f}GB | LoRA: {est['lora_mb']/1024:.1f}GB | 激活: {est['activation_mb']/1024:.1f}GB",
        f"   梯度: {est['gradient_mb']/1024:.1f}GB | 优化器: {est['optimizer_mb']/1024:.1f}GB",
        f"   总计: ~{est['total_gb']}GB / {vram_gb:.0f}GB {bar} {status}",
    ]

    # 训练时间预测
    try:
        from core.simulation.estimator import estimate_training
        from core.utils import guess_model_size
        model_b = guess_model_size(model_id)
        time_est = estimate_training(
            model_params_b=model_b,
            dataset_samples=1000,  # 默认假设 1000 条
            avg_seq_len=int(seq_len or 2048) // 2,
            max_seq_len=int(seq_len or 2048),
            batch_size=int(batch_size or 1),
            rank=int(rank_val or 64),
            use_qlora=bool(qlora_on),
        )
        if time_est and time_est.time_human:
            lines.append(f"   ⏱️ 预估训练时间(1000条): ~{time_est.time_human} ({time_est.steps_total} 步)")
    except Exception:
        pass

    return "\n".join(lines)

# ---------------------------
# UI
# ---------------------------

def build_app():
    gpu = _get_gpu_cached()
    if gpu:
        _dev_type = gpu.get("device_type", "cpu")
        _dev_count = gpu.get("device_count", 1)
        _name = gpu.get("name", "N/A")
        _vram = gpu.get("vram_mb", 0)
        if _dev_count > 1:
            _total_vram = gpu.get("total_vram_mb", _vram * _dev_count)
            env_line = f"GPU: {_name} × {_dev_count} (总 VRAM={_total_vram // 1024}GB)"
        elif _dev_type == "npu":
            env_line = f"NPU: {_name} VRAM={_vram}MB"
        elif _dev_type == "mps":
            env_line = f"Apple MPS (统一内存)"
        else:
            env_line = f"GPU: {_name} VRAM={_vram}MB"
    else:
        env_line = "GPU: N/A"
    with gr.Blocks(title=UI_TITLE) as demo:
        gr.Markdown(
            UI_HEADER
            + "**数据** → **训练** → **编辑** → **评测** → **部署**　|　"
            + f"`{env_line}`"
        )

        # ================================================================
        #  Tab 1: 📊 数据
        # ================================================================
        with gr.Tab("📊 数据") as tab_data:
            gr.Markdown("上传训练数据，支持 JSONL / JSON / CSV / TXT / Parquet 格式。")

            # ---- 上传与预览 ----
            with gr.Accordion("📁 上传与管理", open=True):
                with gr.Row():
                    file_in = gr.File(label="上传数据集（支持多文件：jsonl/json/csv/txt/parquet）", file_count="multiple", scale=3)
                    orig_name = gr.Textbox(label="重命名（可选）", placeholder="同名覆盖", scale=1)
                with gr.Row():
                    btn_up = gr.Button("📤 上传", variant="primary")
                    btn_refresh_ds = gr.Button("🔄 刷新列表")
                    btn_delete_ds = gr.Button("🗑️ 删除", variant="stop")
                    up_msg = gr.Textbox(label="状态", interactive=False, scale=2)
                ds_list = _STARTUP_DS_LIST
                ds_choice = gr.Dropdown(label="数据集列表", choices=ds_list, value=ds_list[0] if ds_list else None)
                with gr.Row():
                    prev_rows = gr.Slider(1, 200, value=50, step=1, label="预览行数", scale=1)
                    btn_prev = gr.Button("👁 预览")
                    btn_dedup = gr.Button("🧹 快速去重")
                prev_out = gr.JSON(label="预览内容")
                btn_prev.click(data_preview, [ds_choice, prev_rows], prev_out)
                btn_dedup.click(data_clean_dedup, ds_choice, up_msg)

            # ---- 格式转换 ----
            with gr.Accordion("🔄 格式转换", open=False):
                gr.Markdown("将数据集在 Alpaca / ShareGPT / OpenAI / 纯文本格式之间互转")
                with gr.Row():
                    cvt_src = gr.Dropdown(label="源数据集", choices=_STARTUP_DS_LIST, scale=2)
                    cvt_target = gr.Dropdown(label="目标格式", choices=["alpaca", "sharegpt", "openai", "text"], value="alpaca", scale=1)
                    btn_cvt = gr.Button("转换", variant="primary", scale=1)
                cvt_msg = gr.Textbox(label="状态", interactive=False)
                def _do_convert(src, tgt):
                    if not src: return "请选择数据集"
                    try:
                        out = dm.convert_format(src, tgt)
                        return f"✅ 转换完成: {out}"
                    except Exception as e: return f"❌ 转换失败: {e}"
                btn_cvt.click(_do_convert, [cvt_src, cvt_target], cvt_msg)

            # ---- 数据处理工具 ----
            with gr.Accordion("🛠️ 清洗 / 分割 / 质量分析", open=False):
                with gr.Row():
                    clean_src = gr.Dropdown(label="选择数据集", choices=_STARTUP_DS_LIST, scale=2)
                    clean_dedup = gr.Checkbox(label="去重", value=True)
                    clean_empty = gr.Checkbox(label="去空", value=True)
                    clean_min_len = gr.Number(label="最小长度", value=10, precision=0)
                    btn_clean = gr.Button("🧹 清洗", variant="primary", scale=1)
                with gr.Row():
                    split_src = gr.Dropdown(label="分割数据集", choices=_STARTUP_DS_LIST, scale=2)
                    split_ratio = gr.Slider(0.5, 0.99, value=0.9, step=0.01, label="训练集比例", scale=1)
                    btn_split = gr.Button("✂️ 分割", variant="primary", scale=1)
                with gr.Row():
                    qa_src = gr.Dropdown(label="分析数据集", choices=_STARTUP_DS_LIST, scale=2)
                    btn_qa = gr.Button("🔬 质量分析", variant="primary", scale=1)
                data_tool_msg = gr.Textbox(label="状态", interactive=False)
                qa_report = gr.Textbox(label="质量报告", interactive=False, lines=10, visible=False)
                def _do_clean(src, dedup, empty, min_len):
                    if not src: return "请选择数据集", gr.update()
                    try:
                        out = dm.clean_dataset(src, {"deduplicate": dedup, "remove_empty": empty, "min_length": int(min_len)})
                        return f"✅ 清洗完成: {out}", gr.update()
                    except Exception as e: return f"❌ {e}", gr.update()
                btn_clean.click(_do_clean, [clean_src, clean_dedup, clean_empty, clean_min_len], [data_tool_msg, qa_report])
                def _do_split(src, ratio):
                    if not src: return "请选择数据集", gr.update()
                    try:
                        result = dm.split_dataset(src, float(ratio))
                        return f"✅ 分割完成: train={result['train']}, val={result['val']}", gr.update()
                    except Exception as e: return f"❌ {e}", gr.update()
                btn_split.click(_do_split, [split_src, split_ratio], [data_tool_msg, qa_report])
                def _do_quality_analysis(src):
                    if not src: return "请选择数据集", gr.update(visible=False)
                    try:
                        from core.simulation.estimator import DatasetAnalyzer
                        analyzer = DatasetAnalyzer()
                        ds_path = str(DATASETS_DIR / src)
                        report = analyzer.analyze(ds_path)
                        return "✅ 分析完成", gr.update(visible=True, value=analyzer.format_report(report))
                    except Exception as e: return f"❌ {e}", gr.update(visible=False)
                btn_qa.click(_do_quality_analysis, qa_src, [data_tool_msg, qa_report])

            # ---- AI 合成 ----
            with gr.Accordion("🤖 AI 数据合成（Ollama / OpenAI）", open=False):
                with gr.Row():
                    synth_topic = gr.Textbox(label="主题", placeholder="例如：Python编程、客服对话", scale=2)
                    synth_count = gr.Number(label="数量", value=50, precision=0, scale=1)
                    synth_fmt = gr.Dropdown(label="格式", choices=["alpaca", "sharegpt"], value="alpaca", scale=1)
                with gr.Row():
                    synth_api = gr.Dropdown(label="API类型", choices=["ollama", "openai"], value="ollama", scale=1)
                    synth_base = gr.Textbox(label="API地址", value="http://localhost:11434", scale=2)
                    synth_key = gr.Textbox(label="API Key", value="", placeholder="OpenAI 需要", scale=1)
                    synth_model = gr.Textbox(label="模型", value="qwen2.5:7b", scale=1)
                btn_synth = gr.Button("🧬 开始生成", variant="primary")
                synth_task_id = gr.Textbox(visible=False)
                synth_msg = gr.Textbox(label="状态", interactive=False)
                def _do_synth(topic, count, fmt, api, base, key, model):
                    if not topic: return "", "请输入主题"
                    from core.ai_synthesizer import ai_synthesizer
                    def _run(task):
                        return ai_synthesizer.generate_sft_data(
                            topic=topic, count=int(count), output_format=fmt,
                            api_type=api, api_base=base, api_key=key, model=model, task=task,
                        )
                    tid = task_queue.submit(f"AI合成: {topic}", _run)
                    return tid, f"✅ 任务已提交: {tid}"
                btn_synth.click(_do_synth, [synth_topic, synth_count, synth_fmt, synth_api, synth_base, synth_key, synth_model], [synth_task_id, synth_msg])

            # 切换到数据 Tab 时刷新所有 dropdown
            def _refresh_data_tab():
                _invalidate_ds_cache()
                new = _list_datasets()
                return gr.update(choices=new), gr.update(choices=new), gr.update(choices=new), gr.update(choices=new), gr.update(choices=new)
            tab_data.select(_refresh_data_tab, None, [cvt_src, clean_src, split_src, ds_choice, qa_src])

        # ================================================================
        #  Tab 2: 🔥 训练
        # ================================================================
        with gr.Tab("🔥 训练") as tab_train:

            gr.Markdown(
                "选择训练模式，填写参数后一键开始。提交后在「📋 任务中心」查看实时进度。"
            )

            with gr.Accordion("🔥 SFT / DPO 标准训练", open=True):
                gr.Markdown("用数据集直接微调模型。适合指令微调、风格模仿、领域适配。")

                # ---- 模型选择 ----
                with gr.Accordion("📦 选择基座模型", open=True):
                    with gr.Row():
                        model_mode = gr.Radio(
                            label="模型来源",
                            choices=["HF 模型ID", "本地模型路径（继续训练）", "上传 GGUF（仅聊天测试/部署）"],
                            value="HF 模型ID", scale=3,
                        )
                        btn_scan_models = gr.Button("🔄 扫描本机", scale=1)

                    # 推荐模型 — 按 VRAM 分组
                    gpu = _get_gpu_cached()
                    _user_vram_mb = gpu.get("vram_mb", 0) if gpu else 0
                    _user_vram_gb = _user_vram_mb / 1024

                    _rec_models_raw = [
                        ("Qwen/Qwen2.5-0.5B-Instruct",                "Qwen2.5-0.5B",   0.5, 2,  "入门首选，中英双语"),
                        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "DS-R1-1.5B",      1.5, 4,  "推理能力强"),
                        ("Qwen/Qwen2.5-1.5B-Instruct",                "Qwen2.5-1.5B",   1.5, 4,  "轻量中英双语"),
                        ("google/gemma-2-2b-it",                       "Gemma-2-2B",      2,   5,  "Google，英文强"),
                        ("Qwen/Qwen2.5-3B-Instruct",                  "Qwen2.5-3B",     3,   6,  "性价比之选"),
                        ("microsoft/Phi-3.5-mini-instruct",            "Phi-3.5-3.8B",   3.8, 7,  "微软，推理好"),
                        ("Qwen/Qwen2.5-7B-Instruct",                  "Qwen2.5-7B",     7,   10, "主力模型"),
                        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",   "DS-R1-7B",        7,   10, "推理+代码"),
                        ("meta-llama/Llama-3.1-8B-Instruct",          "Llama-3.1-8B",   8,   12, "Meta 全能"),
                        ("Qwen/Qwen2.5-14B-Instruct",                 "Qwen2.5-14B",    14,  20, "大参数高质量"),
                    ]
                    _recommended = []
                    last_tier = 0
                    for hf_id_str, short_name, params_b, min_vram, desc in _rec_models_raw:
                        if min_vram <= 4 and last_tier < 4:
                            _recommended.append("── 🟢 ≤4GB VRAM（入门显卡）──")
                            last_tier = 4
                        elif min_vram <= 8 and last_tier < 8:
                            _recommended.append("── 🟢 ≤8GB VRAM ──")
                            last_tier = 8
                        elif min_vram <= 12 and last_tier < 12:
                            _recommended.append("── 🟡 ≤12GB VRAM ──")
                            last_tier = 12
                        elif min_vram <= 16 and last_tier < 16:
                            _recommended.append("── 🟡 ≤16GB VRAM ──")
                            last_tier = 16
                        elif min_vram <= 24 and last_tier < 24:
                            _recommended.append("── 🔴 ≤24GB VRAM（高端显卡）──")
                            last_tier = 24
                        can_run = "✅" if _user_vram_gb >= min_vram else "⚠️"
                        qlora_note = " [需QLoRA]" if (_user_vram_gb > 0 and _user_vram_gb < min_vram and _user_vram_gb >= min_vram * 0.6) else ""
                        _recommended.append(f"{can_run} {short_name} ({params_b}B, ≥{min_vram}GB{qlora_note}) — {desc} | {hf_id_str}")

                    rec_models = gr.Dropdown(label="推荐模型（点击选择）", choices=_recommended, value=None,
                                             info=f"你的显卡: {gpu.get('name','未检测到')} ({_user_vram_gb:.0f}GB)" if gpu else "未检测到 GPU")
                    with gr.Row():
                        hf_cached = gr.Dropdown(label="本机已有 HF 模型", choices=[], value=None, scale=2)
                        hf_id = gr.Textbox(
                            label="HF 模型ID / 路径",
                            value=config.get("default_base_model", ""),
                            placeholder="例: Qwen/Qwen2.5-7B-Instruct", scale=3,
                        )
                        btn_download = gr.Button("⬇️ 下载", scale=1)

                    def _on_rec_select(choice):
                        if not choice or choice.startswith("──"):
                            return gr.update()
                        if " | " in choice:
                            return gr.update(value=choice.split(" | ")[-1].strip())
                        return gr.update(value=choice.split(" (")[0].lstrip("✅⚠️ ").strip())
                    rec_models.change(_on_rec_select, rec_models, hf_id)

                    download_msg = gr.Textbox(label="下载状态", interactive=False, visible=False)
                    def _download_model(model_id):
                        if not model_id or not model_id.strip():
                            return gr.update(visible=True, value="❌ 请输入模型ID")
                        try:
                            def _run(task): return model_manager.download_model(model_id.strip(), task=task)
                            tid = task_queue.submit(f"下载: {model_id.strip()}", _run)
                            return gr.update(visible=True, value=f"✅ 下载任务已提交: {tid}")
                        except Exception as e:
                            return gr.update(visible=True, value=f"❌ 提交失败: {e}")
                    btn_download.click(_download_model, hf_id, download_msg)

                    with gr.Row():
                        local_model_path = gr.Textbox(label="本地模型路径", value="", visible=False, scale=2)
                        local_model_picker = gr.Dropdown(
                            label="快速选择已有模型", choices=_list_all_model_dirs(), value=None, visible=False, scale=2,
                        )
                        gguf_upload = gr.File(label="上传 GGUF", file_types=[".gguf"], visible=False, scale=2)

                    def _on_model_mode_change(mode: str):
                        mode = (mode or "").strip()
                        is_hf = mode.startswith("HF")
                        is_local = ("本地" in mode)
                        is_gguf = ("GGUF" in mode)
                        return (gr.update(visible=is_hf), gr.update(visible=is_hf),
                                gr.update(visible=is_local), gr.update(visible=is_local,
                                    choices=_list_all_model_dirs() if is_local else []),
                                gr.update(visible=is_gguf))
                    model_mode.change(_on_model_mode_change, model_mode,
                                      [hf_cached, hf_id, local_model_path, local_model_picker, gguf_upload])
                    def _pick_local_model(name):
                        if not name: return gr.update()
                        clean = _extract_model_name_from_label(name)
                        return gr.update(value=str(LORAS_DIR / clean))
                    local_model_picker.change(_pick_local_model, local_model_picker, local_model_path)
                    btn_scan_models.click(fn=_refresh_local_hf_model_ids, inputs=None, outputs=hf_cached, show_progress="full")
                    hf_cached.change(lambda x: gr.update(value=x), hf_cached, hf_id)

                # ---- 数据集选择 ----
                with gr.Accordion("📂 选择数据集", open=True):
                    with gr.Row():
                        dataset = gr.Dropdown(label="数据集（可多选）", choices=_STARTUP_DS_LIST, multiselect=True, scale=2,
                                              info="在「数据工坊」上传或生成")
                        btn_refresh_train_ds = gr.Button("🔄 刷新", scale=1)
                    ds_info_display = gr.Textbox(label="📊 数据集信息", interactive=False, value="", lines=2, visible=False)

                    def _show_ds_info(ds_names):
                        if not ds_names:
                            return gr.update(visible=False, value="")
                        info_parts = []
                        total_rows = 0
                        for ds_name in (ds_names if isinstance(ds_names, list) else [ds_names]):
                            ds_path = DATASETS_DIR / ds_name
                            if not ds_path.exists(): continue
                            try:
                                if ds_path.is_dir():
                                    for f in list(ds_path.glob("*.json")) + list(ds_path.glob("*.jsonl")):
                                        if f.suffix == ".jsonl":
                                            count = sum(1 for line in f.read_text(encoding="utf-8").strip().splitlines() if line.strip())
                                        else:
                                            data = json.loads(f.read_text(encoding="utf-8"))
                                            count = len(data) if isinstance(data, list) else 1
                                        total_rows += count
                                        info_parts.append(f"  {ds_name}/{f.name}: {count} 条")
                                elif ds_path.is_file():
                                    text = ds_path.read_text(encoding="utf-8")
                                    if ds_path.suffix == ".jsonl":
                                        count = sum(1 for line in text.strip().splitlines() if line.strip())
                                    elif ds_path.suffix == ".json":
                                        data = json.loads(text)
                                        count = len(data) if isinstance(data, list) else 1
                                    else:
                                        count = sum(1 for line in text.strip().splitlines() if line.strip())
                                    total_rows += count
                                    info_parts.append(f"  {ds_name}: {count} 条")
                            except Exception:
                                info_parts.append(f"  {ds_name}: 读取失败")
                        if not info_parts:
                            return gr.update(visible=False, value="")
                        return gr.update(visible=True, value=f"共 {total_rows} 条\n" + "\n".join(info_parts))
                    dataset.change(_show_ds_info, dataset, ds_info_display)

                    def _refresh_all_ds():
                        _invalidate_ds_cache()
                        return gr.update(choices=_list_datasets())
                    btn_refresh_train_ds.click(_refresh_all_ds, None, dataset, show_progress="minimal")

                    # ── 数据质量面板 ──
                    btn_up.click(data_upload, [file_in, orig_name], [ds_choice, dataset, up_msg])
                    btn_delete_ds.click(data_delete, ds_choice, [ds_choice, dataset, up_msg])
                    def _refresh_both_ds():
                        _invalidate_ds_cache()
                        new = _list_datasets()
                        return gr.update(choices=new), gr.update(choices=new)
                    btn_refresh_ds.click(_refresh_both_ds, None, [ds_choice, dataset])

                # ---- 训练参数 ----
                with gr.Accordion("⚙️ 训练参数", open=True):
                    preset_mode = gr.Radio(
                        label="训练预设",
                        choices=[
                            "🔍 自动检测（根据显卡）",
                            "⚡ 快速冒烟（5-15分钟）",
                            "📈 稳定高质量（推荐）",
                            "💾 低显存模式（8GB以下）",
                        ],
                        value="🔍 自动检测（根据显卡）",
                    )
                    with gr.Row():
                        method = gr.Dropdown(
                            label="训练方法", choices=_STARTUP_METHODS, value="SFT", scale=1,
                            info="SFT=指令微调 | DPO=偏好对齐 | ORPO=无需奖励模型 | KTO=二元反馈",
                        )
                        output_name = gr.Textbox(label="输出名", value="mylora", scale=1,
                                                  info="保存到 data/loras/{输出名}")
                    with gr.Row():
                        lr = gr.Number(label="学习率", value=2e-4, info="通常 1e-4~5e-4")
                        batch = gr.Number(label="批大小", value=1, precision=0, info="显存不够就设 1")
                        epochs = gr.Number(label="训练轮数", value=1.0, info="<1k条用3轮，>5k条用1轮")
                        max_seq = gr.Number(label="最大序列长度", value=2048, precision=0, info="512/1024/2048/4096")
                    with gr.Row():
                        rank = gr.Number(label="LoRA Rank", value=64, precision=0, info="16=轻量 32=平衡 64=高质量")
                        ga_steps = gr.Number(label="梯度累积", value=4, precision=0, info="等效批大小=批大小×此值")
                        qlora = gr.Checkbox(label="QLoRA（4bit量化）", value=False, info="省约60%显存")
                        warmup_ratio = gr.Number(label="预热比例", value=0.05, info="学习率从0缓慢上升")

                    # ── v3.0 质量优化控制 ──
                    with gr.Accordion("🎯 v3.0 质量增强（默认全开）", open=False):
                        gr.Markdown(
                            "这些特性经论文验证，默认全部启用。"
                            " 如果遇到兼容性问题可逐个关闭。"
                        )
                        with gr.Row():
                            use_dora = gr.Checkbox(
                                label="DoRA（Weight-Decomposed LoRA）",
                                value=True, scale=1,
                                info="同 rank 下一致优于标准 LoRA（PEFT ≥0.10）")
                            use_rslora = gr.Checkbox(
                                label="rsLoRA（Rank-Stabilized）",
                                value=True, scale=1,
                                info="高 rank 更稳定，自动 scaling（PEFT ≥0.9）")
                        with gr.Row():
                            use_packing = gr.Checkbox(
                                label="📦 Sample Packing",
                                value=True, scale=1,
                                info="短样本拼接，训练效率翻 2-3 倍")
                            auto_clean = gr.Checkbox(
                                label="🧹 训练前自动清洗",
                                value=True, scale=1,
                                info="自动去重+过滤空答案（数据质量 > 一切训练技巧）")
                        with gr.Row():
                            label_smoothing = gr.Number(
                                label="Label Smoothing",
                                value=0.1, scale=1,
                                info="0=关闭 0.1=推荐 0.2=强正则化")
                            neftune_alpha = gr.Number(
                                label="NEFTune α",
                                value=5.0, scale=1,
                                info="噪声嵌入强度，0=关闭，5=推荐（+25% MT-Bench）")

                    # ── 🧠 MoLoRA → 已整合到差分神经元系统 ──
                    with gr.Accordion("🧠 MoLoRA 多专家训练", open=False):
                        gr.Markdown(
                            "**快捷入口**: 在当前训练中启用 MoLoRA 多专家模式。\n"
                            "完整管理请前往 **模型编辑 Tab → 🧠 差分神经元系统**。"
                        )
                        use_molora = gr.Checkbox(
                            label="启用 MoLoRA", value=False,
                            info="替代标准 LoRA, 多专家 + 内置门控")
                        with gr.Row():
                            molora_n_experts = gr.Slider(2, 8, value=4, step=1, label="专家数")
                            molora_top_k = gr.Slider(1, 4, value=2, step=1, label="Top-K")
                        molora_labels = gr.Textbox(
                            label="专家标签（逗号分隔）",
                            placeholder="医学, 法律, 代码, 通用")
                    vram_hint = gr.Markdown(value="", visible=False)
                    def _vram_estimate_md(model_id_val, qlora_on, rank_val, seq_len, batch_size):
                        text = _vram_estimate_detailed(model_id_val, qlora_on, rank_val, seq_len, batch_size)
                        if not text or text.startswith("💡"):
                            return gr.update(visible=False, value="")
                        return gr.update(visible=True, value=f"```\n{text}\n```")
                    for inp in [qlora, rank, max_seq, batch]:
                        inp.change(_vram_estimate_md, [hf_id, qlora, rank, max_seq, batch], vram_hint)
                    hf_id.change(_vram_estimate_md, [hf_id, qlora, rank, max_seq, batch], vram_hint)

                    # ── 智能推荐 ──
                    with gr.Row():
                        btn_smart = gr.Button("🎯 智能推荐参数", scale=1)
                        smart_result = gr.Markdown(value="", visible=False)

                    def _smart_recommend(model_id_val, ds_names):
                        if not model_id_val:
                            return (gr.update(visible=True, value="❌ 请先选择模型"),
                                    gr.update(), gr.update(), gr.update(), gr.update(),
                                    gr.update(), gr.update(), gr.update(), gr.update())
                        from core.smart_params import recommend_params, format_recommendation_markdown
                        gpu = _get_gpu_cached()
                        vram = gpu.get("vram_mb", 0) if gpu else 0
                        rec = recommend_params(model_id_val, ds_names, vram)
                        md = format_recommendation_markdown(rec)
                        return (
                            gr.update(visible=True, value=md),
                            gr.update(value=rec.get("lr", 2e-4)),
                            gr.update(value=int(rec.get("batch_size", 1))),
                            gr.update(value=rec.get("epochs", 3.0)),
                            gr.update(value=int(rec.get("max_seq_len", 2048))),
                            gr.update(value=rec.get("use_qlora", False)),
                            gr.update(value=int(rec.get("rank", 64))),
                            gr.update(value=int(rec.get("gradient_accumulation_steps", 4))),
                            gr.update(value=rec.get("warmup_ratio", 0.05)),
                        )
                    btn_smart.click(_smart_recommend, [hf_id, dataset],
                                    [smart_result, lr, batch, epochs, max_seq, qlora, rank, ga_steps, warmup_ratio])

                # ---- 训练后处理 ----
                # 自动输出名
                def _auto_output_name(model_id_val, method_val):
                    if not model_id_val:
                        return gr.update()
                    import re as _re
                    short = model_id_val.split("/")[-1].split("-Instruct")[0].split("-instruct")[0]
                    short = _re.sub(r'[^a-zA-Z0-9._-]', '', short)[:30]
                    method_short = (method_val or "sft").lower()
                    ts = time.strftime("%m%d")
                    return gr.update(value=f"{short}_{method_short}_{ts}")
                hf_id.change(_auto_output_name, [hf_id, method], output_name)
                method.change(_auto_output_name, [hf_id, method], output_name)

                def _on_preset_change(pn):
                    _lr,_b,_e,_ms,_q,_r,_ga = _apply_train_preset(pn)
                    return (gr.update(value=_lr), gr.update(value=int(_b)), gr.update(value=_e),
                            gr.update(value=int(_ms)), gr.update(value=_q),
                            gr.update(value=int(_r)), gr.update(value=int(_ga)))
                preset_mode.change(_on_preset_change, preset_mode, [lr, batch, epochs, max_seq, qlora, rank, ga_steps])

                def _smart_epochs_hint(ds_names):
                    if not ds_names: return gr.update()
                    total = 0
                    for ds_name in (ds_names if isinstance(ds_names, list) else [ds_names]):
                        ds_path = DATASETS_DIR / ds_name
                        try:
                            if ds_path.is_file():
                                if ds_path.suffix == ".jsonl":
                                    total += sum(1 for l in ds_path.read_text(encoding="utf-8").strip().splitlines() if l.strip())
                                elif ds_path.suffix == ".json":
                                    data = json.loads(ds_path.read_text(encoding="utf-8"))
                                    total += len(data) if isinstance(data, list) else 1
                            elif ds_path.is_dir():
                                for f in list(ds_path.glob("*.jsonl")) + list(ds_path.glob("*.json")):
                                    if f.suffix == ".jsonl":
                                        total += sum(1 for l in f.read_text(encoding="utf-8").strip().splitlines() if l.strip())
                                    else:
                                        data = json.loads(f.read_text(encoding="utf-8"))
                                        total += len(data) if isinstance(data, list) else 1
                        except Exception: pass
                    if total == 0: return gr.update()
                    if total < 500: return gr.update(value=5.0)
                    elif total < 2000: return gr.update(value=3.0)
                    elif total < 10000: return gr.update(value=1.0)
                    else: return gr.update(value=0.5)
                dataset.change(_smart_epochs_hint, dataset, epochs)

                # ---- 启动 ----
                with gr.Row():
                    auto_merge = gr.Checkbox(label="训练后自动合并 LoRA", value=False, scale=1)
                    export_gguf = gr.Dropdown(label="训练后导出 GGUF", 
                        choices=["", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"],
                        value="", scale=1, info="留空=不导出")
                with gr.Row():
                    recipe_name = gr.Textbox(label="保存为配方（可选）", placeholder="方便下次一键加载", scale=2)
                    btn_train = gr.Button("🚀 开始训练", variant="primary", scale=1, size="lg")
                train_task_id = gr.Textbox(visible=False)
                train_msg = gr.Textbox(label="状态", interactive=False)

                btn_train.click(
                    train_submit_ui,
                    [model_mode, hf_id, local_model_path, dataset, method, output_name, lr, batch, epochs, max_seq, qlora, recipe_name, rank, ga_steps, warmup_ratio, auto_merge, export_gguf, use_dora, use_rslora, use_packing, auto_clean, label_smoothing, neftune_alpha, use_molora, molora_n_experts, molora_top_k, molora_labels],
                    [train_task_id, train_msg],
                )

            # ──────────────────────────────────────────────────

            # ──────────────────────────────────────────────────
            # 模式 B: 知识蒸馏（本地大模型教小模型）
            # ──────────────────────────────────────────────────
            # ──────────────────────────────────────────────────
            # 模式 C: API 教师蒸馏（含做题学习）
            # ──────────────────────────────────────────────────
            with gr.Accordion("🌐 API 教师蒸馏 — 用 GPT/DeepSeek/Claude 教小模型", open=False):
                gr.Markdown(
                    "API 大模型当老师，三种学习方式:\n"
                    "- **📝 生成训练集**: 教师回答所有问题 → SFT 训练学生\n"
                    "- **🎯 做题学习**: 每轮 API 出新题 → 学生答 → 教师改 → 训练 → 下轮用改进后模型（推荐！）\n"
                    "- **🏆 做题+DPO**: 做题学习 + 学生错答vs教师正答 → 偏好对齐"
                )

                with gr.Accordion("🔗 API 教师连接", open=True):
                    with gr.Row():
                        api_endpoint_preset = gr.Dropdown(
                            label="快速选择服务商",
                            choices=["自定义", "DeepSeek", "OpenAI", "Claude (Anthropic)", "本地 vLLM/Ollama"],
                            value="自定义", scale=1, info="自动填充端点和模型",
                        )
                        api_base = gr.Textbox(label="API 端点", value="https://api.openai.com/v1", scale=2)
                        api_key = gr.Textbox(label="API Key", value="", type="password", scale=1)
                    with gr.Row():
                        api_model = gr.Dropdown(label="教师模型",
                                               choices=["gpt-4o-mini", "gpt-4o", "deepseek-chat",
                                                        "deepseek-reasoner", "claude-sonnet-4-20250514",
                                                        "claude-haiku-4-5-20251001", "qwen-plus"],
                                               value="gpt-4o-mini", allow_custom_value=True, scale=2,
                                               info="选择或手动输入")
                        api_workers = gr.Slider(1, 20, value=5, step=1, label="并行线程", scale=1, info="API 并行调用数")
                    api_sys_prompt = gr.Textbox(label="教师 System Prompt（可选）",
                                                 placeholder="例: 你是一位专业的中文写作助手，回答要详细准确。",
                                                 value="", lines=2, info="指导教师回答风格")

                    def _on_endpoint_preset(preset):
                        presets = {
                            "DeepSeek": ("https://api.deepseek.com", "deepseek-chat"),
                            "OpenAI": ("https://api.openai.com/v1", "gpt-4o-mini"),
                            "Claude (Anthropic)": ("https://api.anthropic.com", "claude-sonnet-4-20250514"),
                            "本地 vLLM/Ollama": ("http://localhost:8000/v1", "default"),
                        }
                        if preset in presets:
                            base, model = presets[preset]
                            return gr.update(value=base), gr.update(value=model)
                        return gr.update(), gr.update()
                    api_endpoint_preset.change(_on_endpoint_preset, api_endpoint_preset, [api_base, api_model])

                with gr.Accordion("🎓 学生模型 & 数据", open=True):
                    with gr.Row():
                        api_student = gr.Textbox(label="学生模型（HF ID 或路径）",
                                                  placeholder="例: Qwen/Qwen2.5-1.5B-Instruct", scale=2,
                                                  info="要训练的本地小模型")
                        api_student_picker = gr.Dropdown(label="从本机选择", choices=_list_all_model_dirs(), value=None, scale=1)
                    with gr.Row():
                        api_dataset = gr.Dropdown(label="种子数据 / 题库", choices=_list_datasets(), scale=2,
                                                  info="含 instruction 字段，做题模式用作出题参考主题")
                        api_output = gr.Textbox(label="输出名", value="api_distilled", scale=1)

                    def _pick_api_student(name):
                        if not name: return gr.update()
                        clean = _extract_model_name_from_label(name)
                        return gr.update(value=str(LORAS_DIR / clean))
                    api_student_picker.change(_pick_api_student, api_student_picker, api_student)

                with gr.Accordion("📚 学习模式", open=True):
                    api_exam_mode = gr.Radio(
                        label="选择模式",
                        choices=[
                            "📝 传统生成（教师回答种子数据中的所有题目）",
                            "🎯 做题学习（每轮API出新题→学生答→教师改→训练→循环）",
                            "🏆 做题+DPO（做题学习 + 偏好对齐，效果最好但更慢）",
                        ],
                        value="📝 传统生成（教师回答种子数据中的所有题目）",
                    )
                    # ---- 传统模式参数 ----
                    api_gen_row = gr.Row(visible=True)
                    with api_gen_row:
                        api_num_resp = gr.Number(label="每条生成次数", value=1, precision=0,
                                                  info="每条 prompt 生成几个教师回复")
                        api_temp = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="教师温度", info="越高越多样")
                        api_top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.05, label="教师 Top-P", info="核采样概率")
                        api_max_tokens = gr.Slider(256, 8192, value=2048, step=256,
                                                    label="教师最大回复长度", info="教师每条回复的 token 上限")
                        api_pipeline = gr.Checkbox(label="🔄 流水线（边生边训）", value=False)
                        api_pipeline_batch = gr.Number(label="批次大小", value=500, precision=0, visible=False)

                    # ---- 做题模式参数 ----
                    api_exam_row = gr.Column(visible=False)
                    with api_exam_row:
                        gr.Markdown("**每轮流程**: API 根据种子数据主题出新题 → 学生用当前模型答题 → 教师批改打分 → 错题加权训练 → 下一轮用改进后的模型继续")
                        with gr.Row():
                            api_exam_rounds = gr.Number(label="训练轮数", value=3, precision=0,
                                                         info="几轮循环（每轮出新题+训练，推荐 3-5）")
                            api_qpr = gr.Number(label="每轮出题数", value=50, precision=0,
                                                  info="API 每轮生成多少道新题（推荐 30-200）")
                            api_weak_focus = gr.Slider(1.0, 5.0, value=2.0, step=0.5,
                                                        label="错题重复倍率",
                                                        info="错题在训练集中出现几倍")

                        with gr.Accordion("🧪 增强模式（从根源提升训练质量）", open=False):
                            gr.Markdown(
                                "**四个增强维度**，可自由组合:\n"
                                "- **CoT 思维链**: 教师回答包含推理过程，学生学会如何思考\n"
                                "- **Best-of-N**: 学生每题生成 N 个答案，教师选优劣，产生更丰富的 DPO 训练对\n"
                                "- **渐进 Rank**: LoRA rank 每轮递增，模型容量随知识增长\n"
                                "- **主题专家**: 按主题训练专门 LoRA，等效 MoE 神经元路由"
                            )
                            with gr.Row():
                                api_cot = gr.Checkbox(label="🧠 CoT 思维链蒸馏",
                                                       value=True,
                                                       info="教师输出完整推理过程（最大提升项）")
                                api_bon = gr.Slider(1, 5, value=1, step=1,
                                                     label="🎯 Best-of-N 采样",
                                                     info="1=关闭，3=推荐（每题3个答案选优劣）")
                            with gr.Row():
                                api_progressive = gr.Checkbox(label="📈 渐进 LoRA Rank",
                                                               value=False,
                                                               info="每轮自动提高 rank（如 32→48→64→96）")
                                api_topic_expert = gr.Checkbox(label="🧩 主题专家 MoLoRA",
                                                                value=False,
                                                                info="按主题训练专门LoRA，等效神经元路由")
                                api_topic_k = gr.Slider(2, 8, value=4, step=1,
                                                         label="专家数量",
                                                         info="主题分类数（推荐 3-5）",
                                                         visible=False)
                            # 联动显示专家数
                            api_topic_expert.change(
                                lambda v: gr.update(visible=v),
                                api_topic_expert, api_topic_k)

                    def _on_exam_mode_change(mode_val):
                        is_exam = "做题" in (mode_val or "")
                        return (
                            gr.update(visible=not is_exam),  # api_gen_row
                            gr.update(visible=is_exam),      # api_exam_row
                            gr.update(visible=not is_exam),  # btn_api_data_only
                        )

                with gr.Accordion("🎛️ 学生训练参数", open=False):
                    with gr.Row():
                        api_lr = gr.Number(label="学习率", value=2e-4, info="通常 1e-4~5e-4")
                        api_batch = gr.Number(label="批大小", value=1, precision=0, info="显存不够设1")
                        api_epochs = gr.Number(label="每轮Epochs", value=3, info="做题模式: 每轮训练几个epoch")
                        api_max_seq = gr.Number(label="最大序列长度", value=2048, precision=0, info="512/1024/2048")
                    with gr.Row():
                        api_rank = gr.Number(label="LoRA Rank", value=64, precision=0, info="16=轻量 64=高质量")
                        api_ga = gr.Number(label="梯度累积", value=4, precision=0, info="等效批大小=批×此值")
                        api_qlora = gr.Checkbox(label="QLoRA（4bit量化）", value=False, info="省约60%显存")

                with gr.Accordion("💪 训练前知识膨胀（可选 — 先长身体再灌知识）", open=False):
                    gr.Markdown(
                        "在 API 蒸馏 / 做题学习**开始之前**，先自动膨胀学生模型的参数量。\n"
                        "更大的模型有更多参数容量来吸收教师知识。膨胀保留原始权重，不是随机初始化。\n\n"
                        "典型用法: 0.5B 模型 → 膨胀到 0.8B → 用 GPT-4 蒸馏 → 质量超越未膨胀的 1B 模型"
                    )
                    api_expand = gr.Checkbox(label="✅ 启用训练前膨胀", value=False,
                        info="勾选后，训练开始时会先自动膨胀学生模型")
                    api_expand_opts = gr.Column(visible=False)
                    with api_expand_opts:
                        with gr.Row():
                            api_expand_method = gr.Dropdown(
                                label="膨胀方式",
                                choices=[
                                    "depth — 加深（复制层，最安全）",
                                    "width — 加宽（扩展hidden_size）",
                                    "depth+width — 同时加深加宽（最强）",
                                ],
                                value="depth — 加深（复制层，最安全）",
                                info="depth 最安全，depth+width 参数量增长最多",
                            )
                        with gr.Row():
                            api_expand_layers = gr.Number(label="新增层数", value=4, precision=0,
                                info="深度膨胀: 增加几层 Transformer (4-8 推荐)")
                            api_expand_hidden = gr.Number(label="目标 Hidden Size", value=0, precision=0,
                                info="宽度膨胀: 目标隐藏维度 (0=不扩宽)")
                        with gr.Row():
                            api_expand_strategy = gr.Dropdown(
                                label="层复制策略",
                                choices=["repeat_middle", "repeat_all", "interleave"],
                                value="repeat_middle",
                                info="repeat_middle 最安全，interleave 更均匀",
                            )
                            api_expand_noise = gr.Number(label="噪声系数", value=0.01,
                                info="打破对称性的噪声 (0.005-0.02)")

                    api_expand.change(
                        lambda v: gr.update(visible=v),
                        api_expand, api_expand_opts,
                    )

                with gr.Accordion("⚗️ 训练后知识浓缩（可选 — 剪枝+自蒸馏+困难精修）", open=False):
                    gr.Markdown(
                        "训练完成后自动执行「锻压」: 砍掉摸鱼的参数 → 用软标签浓缩暗知识 → 困难样本精修。\n"
                        "效果: 训练后的模型虽大但冗余多，浓缩后**同样大小的模型知识密度翻倍**。\n\n"
                        "🔥 **膨胀 + 浓缩 = 完整锻压**: 先长身体灌知识，再压缩精华，1B 可顶 3-10B。"
                    )
                    api_condense = gr.Checkbox(label="✅ 启用训练后浓缩", value=False,
                        info="训练完自动剪枝+自蒸馏+困难精修")
                    api_condense_opts = gr.Column(visible=False)
                    with api_condense_opts:
                        with gr.Row():
                            api_cond_cycles = gr.Number(label="浓缩轮数", value=1, precision=0,
                                info="1=单次淬火, 2-3=深度锻压（每轮加大力度）")
                            api_cond_temp = gr.Slider(1.0, 10.0, value=3.0, step=0.5,
                                label="自蒸馏温度",
                                info="越高→传递越多暗知识（2-5推荐）")
                        with gr.Row():
                            api_cond_heads = gr.Slider(0.0, 0.5, value=0.2, step=0.05,
                                label="Head 剪枝比例",
                                info="砍掉多少 attention head（0.2=砍20%）")
                            api_cond_neurons = gr.Slider(0.0, 0.5, value=0.15, step=0.05,
                                label="Neuron 剪枝比例",
                                info="砍掉多少 MLP neuron（0.15=砍15%）")
                        with gr.Row():
                            api_cond_layers = gr.Number(label="剪层数", value=0, precision=0,
                                info="剪掉几个 Transformer 层（0=不剪层，激进时可设1-2）")
                            api_cond_hard = gr.Slider(0.1, 0.5, value=0.3, step=0.05,
                                label="困难样本比例",
                                info="挖掘 top 多少最难样本做精修（0.3=最难30%）")
                        with gr.Row():
                            api_cond_distill = gr.Checkbox(label="自蒸馏", value=True,
                                info="剪枝后用原模型软标签再蒸馏")
                            api_cond_mine = gr.Checkbox(label="困难精修", value=True,
                                info="挖掘薄弱样本集中再训练")

                    api_condense.change(
                        lambda v: gr.update(visible=v),
                        api_condense, api_condense_opts,
                    )

                with gr.Row():
                    btn_api_kd = gr.Button("🚀 开始学习", variant="primary", scale=2)
                    btn_api_data_only = gr.Button("📡 仅生成数据（不训练）", variant="secondary", scale=1)
                api_kd_task_id = gr.Textbox(visible=False)
                api_kd_msg = gr.Textbox(label="状态", interactive=False)

                with gr.Accordion("📊 做题成绩图表", open=False, visible=True) as api_chart_acc:
                    api_chart_html = gr.HTML(value="<p style='color:#8b949e;text-align:center;padding:20px;'>训练完成后，成绩图表会显示在这里。<br>也可在 <code>data/datasets/{输出名}_exam/exam_chart.html</code> 查看完整报告。</p>")

                # ---- 绑定事件 ----
                api_exam_mode.change(
                    _on_exam_mode_change, api_exam_mode,
                    [api_gen_row, api_exam_row, btn_api_data_only],
                )

                def _parse_exam_mode(radio_val):
                    if "做题+DPO" in (radio_val or ""):
                        return "exam_dpo"
                    elif "做题学习" in (radio_val or ""):
                        return "exam"
                    return "generate"

                def api_kd_submit(base, key, model, student, student_pick, ds, out,
                                  exam_mode_radio, num_resp, temp, top_p_val, max_tok_val,
                                  pipeline, pipeline_batch,
                                  exam_rounds, qpr, weak_focus,
                                  cot_val, bon_val, progressive_val, topic_expert_val, topic_k_val,
                                  sys_prompt, lr_val, batch_val, ep, max_seq_val,
                                  rank_val, ga_val, qlora_val,
                                  workers_val,
                                  expand_enabled, expand_method_val, expand_layers_val,
                                  expand_hidden_val, expand_strat_val, expand_noise_val,
                                  condense_enabled, cond_cycles, cond_temp,
                                  cond_heads, cond_neurons, cond_layers, cond_hard,
                                  cond_distill, cond_mine,
                                  data_only=False):
                    if not key:
                        return "", "❌ 请填写 API Key"
                    student_val = student or ""
                    if not student_val and not data_only:
                        return "", "❌ 请填写学生模型"
                    if not ds:
                        return "", "❌ 请选择种子数据 / 题库"
                    exam_mode = _parse_exam_mode(exam_mode_radio)
                    from core.distiller import APITeacherConfig, api_teacher

                    # 解析膨胀方式
                    _exp_method = (expand_method_val or "").split(" — ")[0].strip() if expand_method_val else "depth"

                    cfg = APITeacherConfig(
                        api_base=base or "https://api.openai.com/v1",
                        api_key=key,
                        teacher_model_name=model or "gpt-4o-mini",
                        student_model=student_val,
                        dataset_path=ds,
                        output_name=out or "api_distilled",
                        temperature=float(temp or 0.7),
                        top_p=float(top_p_val if top_p_val is not None else 0.95),
                        max_tokens=int(max_tok_val or 2048),
                        system_prompt=sys_prompt or "",
                        num_responses=int(num_resp or 1),
                        lr=float(lr_val or 2e-4),
                        batch_size=int(batch_val or 1),
                        epochs=float(ep or 3),
                        max_seq_len=int(max_seq_val or 2048),
                        rank=int(rank_val or 64),
                        gradient_accumulation_steps=int(ga_val or 4),
                        use_qlora=bool(qlora_val),
                        workers=int(workers_val or 5),
                        pipeline_mode=bool(pipeline),
                        pipeline_batch_size=int(pipeline_batch or 500),
                        exam_mode=exam_mode,
                        exam_rounds=int(exam_rounds or 3),
                        weak_focus_ratio=float(weak_focus or 2.0),
                        questions_per_round=int(qpr or 50),
                        # 增强模式
                        cot_distill=bool(cot_val),
                        best_of_n=int(bon_val or 1),
                        progressive_rank=bool(progressive_val),
                        topic_expert=bool(topic_expert_val),
                        topic_expert_k=int(topic_k_val or 4),
                        # 知识膨胀
                        expand_before_train=bool(expand_enabled),
                        expand_method=_exp_method,
                        expand_extra_layers=int(expand_layers_val or 4),
                        expand_target_hidden=int(expand_hidden_val or 0),
                        expand_depth_strategy=expand_strat_val or "repeat_middle",
                        expand_noise=float(expand_noise_val or 0.01),
                        # 训练后浓缩
                        condense_after_train=bool(condense_enabled),
                        condense_cycles=int(cond_cycles or 1),
                        condense_prune_heads=float(cond_heads or 0.2),
                        condense_prune_neurons=float(cond_neurons or 0.15),
                        condense_prune_layers=int(cond_layers or 0),
                        condense_self_distill=bool(cond_distill),
                        condense_distill_temp=float(cond_temp or 3.0),
                        condense_hard_mine=bool(cond_mine),
                        condense_hard_ratio=float(cond_hard or 0.3),
                    )
                    w = int(workers_val or 5)
                    if data_only:
                        name = f"API 数据生成: {model} (×{w}并行)"
                        def _run(task):
                            result = api_teacher.generate_teacher_data(cfg, task=task)
                            _invalidate_ds_cache()
                            return str(result)
                    elif exam_mode in ("exam", "exam_dpo"):
                        r = int(exam_rounds or 3)
                        q = int(qpr or 50)
                        mode_label = "做题+DPO" if exam_mode == "exam_dpo" else "做题学习"
                        name = f"📝 {mode_label}: {Path(student_val).name} ← {model} | {r}轮×{q}题"
                        def _run(task):
                            result = api_teacher.train_student(cfg, task=task)
                            _invalidate_loras_cache()
                            _invalidate_ds_cache()
                            return result
                    else:
                        mode_label = "流水线" if pipeline else "标准"
                        name = f"API蒸馏[{mode_label}]: {Path(student_val).name} ← {model} (×{w}并行)"
                        def _run(task):
                            result = api_teacher.train_student(cfg, task=task)
                            _invalidate_loras_cache()
                            return result
                    tid = task_queue.submit(name, _run)
                    expand_tag = f" | 💪膨胀({_exp_method})" if expand_enabled else ""
                    condense_tag = f" | ⚗️浓缩(×{int(cond_cycles or 1)})" if condense_enabled else ""
                    if exam_mode in ("exam", "exam_dpo"):
                        enh_tags = []
                        if cot_val: enh_tags.append("CoT")
                        if int(bon_val or 1) > 1: enh_tags.append(f"BoN×{int(bon_val)}")
                        if progressive_val: enh_tags.append("渐进Rank")
                        if topic_expert_val: enh_tags.append(f"MoLoRA×{int(topic_k_val or 4)}")
                        if expand_enabled: enh_tags.append(f"膨胀({_exp_method})")
                        if condense_enabled: enh_tags.append(f"浓缩×{int(cond_cycles or 1)}")
                        enh_str = f" | 增强: {'+'.join(enh_tags)}" if enh_tags else ""
                        return tid, f"✅ 做题学习已启动: {tid} | {int(exam_rounds or 3)}轮 × {int(qpr or 50)}题/轮{enh_str}"
                    return tid, f"✅ 任务已提交: {tid} | {w}线程并行{expand_tag}{condense_tag}"

                _api_inputs = [api_base, api_key, api_model, api_student, api_student_picker,
                               api_dataset, api_output,
                               api_exam_mode, api_num_resp, api_temp, api_top_p, api_max_tokens,
                               api_pipeline, api_pipeline_batch,
                               api_exam_rounds, api_qpr, api_weak_focus,
                               api_cot, api_bon, api_progressive, api_topic_expert, api_topic_k,
                               api_sys_prompt, api_lr, api_batch, api_epochs, api_max_seq,
                               api_rank, api_ga, api_qlora,
                               api_workers,
                               api_expand, api_expand_method, api_expand_layers,
                               api_expand_hidden, api_expand_strategy, api_expand_noise,
                               api_condense, api_cond_cycles, api_cond_temp,
                               api_cond_heads, api_cond_neurons, api_cond_layers, api_cond_hard,
                               api_cond_distill, api_cond_mine]
                btn_api_kd.click(
                    api_kd_submit, _api_inputs, [api_kd_task_id, api_kd_msg],
                )
                btn_api_data_only.click(
                    lambda *args: api_kd_submit(*args, data_only=True),
                    _api_inputs, [api_kd_task_id, api_kd_msg],
                )


            # ──────────────────────────────────────────────────
            # 模式 D: 📖 文档课程训练 — 把一本书教给模型
            # ──────────────────────────────────────────────────
            with gr.Accordion("📖 文档课程训练 — 上传文档 → 自动课程 → 模型学会", open=False):
                gr.Markdown(
                    "上传 PDF / TXT / Markdown / Word 文档，系统自动:\n"
                    "1. 解析文档结构 → 拆分为知识单元\n"
                    "2. 按难度编排课程（基础→进阶→精通）\n"
                    "3. 为每个知识点自动生成问答训练对\n"
                    "4. 分阶段训练，每阶段考试验证掌握程度\n\n"
                    "> 💡 与知识库(RAG)不同：这是把知识**写入模型参数**，模型真正「学会」。"
                )

                with gr.Accordion("📂 文档 & 模型", open=True):
                    with gr.Row():
                        doc_files = gr.File(
                            label="上传文档（支持 PDF/TXT/MD/DOCX，可多选）",
                            file_count="multiple",
                            file_types=[".pdf", ".txt", ".md", ".docx", ".doc", ".json", ".jsonl"],
                            scale=2,
                        )
                    with gr.Row():
                        doc_model_picker = gr.Dropdown(
                            label="学生模型", choices=_list_all_model_dirs(),
                            scale=2, info="选择要教知识的模型",
                        )
                        doc_output_name = gr.Textbox(
                            label="输出名称", placeholder="doc_trained",
                            scale=1,
                        )
                    doc_preview_md = gr.Markdown(visible=False)
                    btn_doc_preview = gr.Button("🔍 预览文档结构", size="sm")

                with gr.Accordion("⚙️ 课程参数", open=False):
                    with gr.Row():
                        doc_stages = gr.Slider(1, 3, value=3, step=1,
                                               label="课程阶段数", info="1=不分阶段, 3=基础→进阶→精通")
                        doc_qa_per_unit = gr.Slider(3, 15, value=5, step=1,
                                                    label="每知识点 QA 数", info="越多学得越深，但更耗时")
                    with gr.Row():
                        doc_use_model_qa = gr.Checkbox(
                            label="用模型生成 QA（质量更高，更耗时 & 显存）",
                            value=False,
                        )
                    with gr.Row():
                        doc_lr = gr.Number(label="学习率", value=2e-4, precision=6)
                        doc_epochs = gr.Number(label="每阶段 Epochs", value=2.0, precision=1)
                        doc_max_seq = gr.Slider(512, 4096, value=2048, step=128, label="Max Seq Len")
                        doc_rank = gr.Slider(8, 128, value=64, step=8, label="LoRA Rank")

                with gr.Row():
                    btn_doc_train = gr.Button("📖 开始课程训练", variant="primary", size="lg")
                doc_task_id = gr.Textbox(visible=False)
                doc_msg = gr.Textbox(label="状态", interactive=False)

                # ── 事件绑定 ──
                def _doc_preview(files):
                    if not files:
                        return gr.update(visible=False, value="")
                    try:
                        from core.doc_curriculum import preview_document
                        lines = []
                        for f in files:
                            fp = f.name if hasattr(f, 'name') else str(f)
                            info = preview_document(fp)
                            lines.append(f"### 📄 {info['file']}")
                            lines.append(f"- **章节**: {info['sections']} | **知识单元**: {info['units']} | **总字数**: {info['total_chars']:,}")
                            lines.append(f"- **难度范围**: {info['difficulty_range']}")
                            lines.append(f"- **关键概念**: {', '.join(info['key_concepts'][:10])}")
                            for sd in info['stage_detail']:
                                lines.append(f"  - {sd['title']}: {sd['units']} 单元 (难度 {sd['avg_difficulty']:.2f})")
                            lines.append("")
                        return gr.update(visible=True, value="\n".join(lines))
                    except Exception as e:
                        return gr.update(visible=True, value=f"❌ 预览失败: {e}")

                btn_doc_preview.click(_doc_preview, [doc_files], [doc_preview_md])

                def _doc_train_submit(model_name, files, output, stages, qa_n,
                                      use_model, lr_val, epochs_val, max_seq_val, rank_val):
                    if not model_name:
                        return "", "❌ 请选择学生模型"
                    if not files:
                        return "", "❌ 请上传至少一个文档"
                    try:
                        model_p = _resolve_model_path(model_name)
                        from core.doc_curriculum import doc_curriculum_submit
                        from core.task_queue import task_queue

                        def _run(task):
                            return doc_curriculum_submit(
                                model_path=model_p,
                                doc_files=files,
                                output_name=str(output or "doc_trained"),
                                num_stages=int(stages),
                                qa_per_unit=int(qa_n),
                                use_model_qa=bool(use_model),
                                lr=float(lr_val),
                                epochs=float(epochs_val),
                                max_seq=int(max_seq_val),
                                rank=int(rank_val),
                                task=task,
                            )
                        tid = task_queue.submit(f"📖 文档课程: {output or 'doc_trained'}", _run)
                        return tid, f"✅ 已提交 | 任务 ID: {tid}"
                    except Exception as e:
                        return "", f"❌ {e}"

                btn_doc_train.click(
                    _doc_train_submit,
                    [doc_model_picker, doc_files, doc_output_name, doc_stages,
                     doc_qa_per_unit, doc_use_model_qa, doc_lr, doc_epochs,
                     doc_max_seq, doc_rank],
                    [doc_task_id, doc_msg],
                )


            # ──────────────────────────────────────────────────
            # 模式 E: 🧬 自我进化训练 — 模型自举迭代提升
            # ──────────────────────────────────────────────────
            with gr.Accordion("🧬 自我进化训练 — 零数据零API，模型自我迭代变强", open=False):
                gr.Markdown(
                    "模型自己给自己出题 → 自己答 → 自己评分 → 筛选最优回答训练自己。\n"
                    "每轮指令自动进化变难，模型能力螺旋上升。\n\n"
                    "> 💡 不需要外部 API，不需要预制数据集，纯本地运算。\n"
                    "> 基于 Evol-Instruct + STaR + ReST 等前沿方法。"
                )

                with gr.Accordion("🎯 模型 & 目标", open=True):
                    with gr.Row():
                        evo_model_picker = gr.Dropdown(
                            label="基座模型", choices=_list_all_model_dirs(),
                            scale=2, info="选择要进化的模型",
                        )
                        evo_output_name = gr.Textbox(
                            label="输出名称", placeholder="evolved_model", scale=1,
                        )
                    evo_topics = gr.Textbox(
                        label="种子主题（逗号分隔）",
                        placeholder="通用知识, 逻辑推理, 创意写作, 数学计算, 代码编程",
                        info="模型将围绕这些主题生成指令并自我训练",
                    )

                with gr.Accordion("⚙️ 进化参数", open=False):
                    with gr.Row():
                        evo_rounds = gr.Slider(1, 10, value=3, step=1,
                                               label="进化轮数", info="越多越强，但越耗时")
                        evo_inst_per_round = gr.Slider(50, 500, value=200, step=50,
                                                       label="每轮指令数")
                    with gr.Row():
                        evo_candidates = gr.Slider(2, 5, value=3, step=1,
                                                   label="每条指令候选数", info="越多评分越准")
                        evo_threshold = gr.Slider(0.3, 0.9, value=0.6, step=0.05,
                                                  label="质量阈值", info="低于此分的回答丢弃")
                    with gr.Row():
                        evo_method = gr.Radio(
                            ["sft", "sft+dpo"], value="sft",
                            label="训练方式", info="sft+dpo: 额外用差回答做偏好对齐",
                        )
                        evo_verify = gr.Checkbox(label="自我验证（模型检查自己的回答）",
                                                 value=True)
                    with gr.Row():
                        evo_lr = gr.Number(label="学习率", value=2e-4, precision=6)
                        evo_epochs = gr.Number(label="每轮 Epochs", value=1.0, precision=1)
                        evo_max_seq = gr.Slider(512, 4096, value=2048, step=128, label="Max Seq")
                        evo_rank = gr.Slider(8, 128, value=64, step=8, label="LoRA Rank")

                with gr.Row():
                    btn_evo_train = gr.Button("🧬 开始进化", variant="primary", size="lg")
                evo_task_id = gr.Textbox(visible=False)
                evo_msg = gr.Textbox(label="状态", interactive=False)

                def _evo_train_submit(model_name, output, topics_str, rounds, inst_n,
                                      cands, threshold, method, verify,
                                      lr_val, epochs_val, max_seq_val, rank_val):
                    if not model_name:
                        return "", "❌ 请选择基座模型"
                    try:
                        model_p = _resolve_model_path(model_name)
                        from core.self_evolve import self_evolve_submit
                        from core.task_queue import task_queue

                        def _run(task):
                            return self_evolve_submit(
                                model_path=model_p,
                                output_name=str(output or "evolved"),
                                seed_topics_str=str(topics_str or ""),
                                num_rounds=int(rounds),
                                instructions_per_round=int(inst_n),
                                candidates=int(cands),
                                quality_threshold=float(threshold),
                                method=str(method),
                                use_verify=bool(verify),
                                lr=float(lr_val),
                                epochs=float(epochs_val),
                                max_seq=int(max_seq_val),
                                rank=int(rank_val),
                                task=task,
                            )
                        tid = task_queue.submit(f"🧬 自我进化: {output or 'evolved'}", _run)
                        return tid, f"✅ 已提交 | 任务 ID: {tid}"
                    except Exception as e:
                        return "", f"❌ {e}"

                btn_evo_train.click(
                    _evo_train_submit,
                    [evo_model_picker, evo_output_name, evo_topics,
                     evo_rounds, evo_inst_per_round, evo_candidates,
                     evo_threshold, evo_method, evo_verify,
                     evo_lr, evo_epochs, evo_max_seq, evo_rank],
                    [evo_task_id, evo_msg],
                )


            # ──────────────────────────────────────────────────
            # 模式 F: 🔮 多模态训练 — 让模型看懂图片/听懂音频
            # ──────────────────────────────────────────────────
            with gr.Accordion("🔮 多模态训练 — 图文理解 / 音文理解 (LLaVA-style)", open=False):
                gr.Markdown(
                    "为文本模型添加视觉/音频理解能力 (LLaVA 架构):\n"
                    "- **冻结**视觉/音频编码器 (SigLIP/CLIP/Whisper)\n"
                    "- **训练**投影层 (MLP bridge) + **LoRA** 微调 LLM\n"
                    "- 8GB 显存可训，两阶段策略\n\n"
                    "> 数据格式: `{\"image\": \"xxx.jpg\", \"conversations\": [{\"from\":\"human\",\"value\":\"<image>描述这张图\"}, ...]}`"
                )
                with gr.Accordion("🎯 模型 & 模态", open=True):
                    with gr.Row():
                        mm_base_model = gr.Dropdown(
                            label="基座文本模型", choices=_list_all_model_dirs(),
                            scale=2, info="要添加视觉/音频能力的文本 LLM",
                        )
                        mm_output = gr.Textbox(label="输出名称", value="multimodal_model", scale=1)
                    with gr.Row():
                        mm_vision = gr.Checkbox(label="👁️ 启用视觉", value=True)
                        mm_audio = gr.Checkbox(label="🔊 启用音频", value=False)
                    with gr.Row():
                        mm_vision_enc = gr.Dropdown(
                            label="视觉编码器",
                            choices=["siglip-base — SigLIP (384d, 推荐)",
                                     "clip-vit-b — CLIP ViT-B/16 (经典)",
                                     "clip-vit-l — CLIP ViT-L/14 (更强)",
                                     "siglip-so400m — SigLIP-SO400M (最强)"],
                            value="siglip-base — SigLIP (384d, 推荐)", scale=2,
                        )
                        mm_audio_enc = gr.Dropdown(
                            label="音频编码器",
                            choices=["whisper-base — Whisper-base (推荐)",
                                     "whisper-small — Whisper-small (更强)"],
                            value="whisper-base — Whisper-base (推荐)", scale=2,
                        )
                with gr.Accordion("📂 训练数据", open=True):
                    mm_dataset = gr.Dropdown(label="多模态数据集", choices=_list_datasets(),
                                              info="需含 image/audio 字段 + conversations")
                    mm_data_info = gr.Markdown("", visible=False)

                with gr.Accordion("⚙️ 训练参数", open=False):
                    with gr.Row():
                        mm_projector = gr.Radio(["linear", "mlp2x", "mlp4x"], value="mlp2x",
                                                 label="投影层类型", info="mlp2x 平衡效果和速度")
                        mm_two_stage = gr.Checkbox(label="两阶段训练（先投影层→再 LoRA+投影层）",
                                                    value=True)
                    with gr.Row():
                        mm_lr = gr.Number(label="投影层学习率", value=1e-3, precision=6)
                        mm_lr_llm = gr.Number(label="LLM LoRA 学习率", value=2e-5, precision=6)
                        mm_rank = gr.Slider(8, 128, value=32, step=8, label="LoRA Rank")
                    with gr.Row():
                        mm_epochs = gr.Number(label="总 Epochs", value=3, precision=1)
                        mm_batch = gr.Number(label="Batch Size", value=1, precision=0)
                        mm_max_seq = gr.Slider(512, 4096, value=2048, step=128, label="Max Seq Len")
                        mm_ga = gr.Number(label="梯度累积", value=8, precision=0)

                    # ── AI 注意力窗口 ──
                    with gr.Accordion("🧠 AI 注意力窗口（控制跨模态注意力范围）", open=False):
                        gr.Markdown(
                            "控制模型处理多模态输入时的注意力机制。\n"
                            "- **视觉 Token 预算**: 每张图片占用的 token 数，越少=越快但细节丢失\n"
                            "- **滑动窗口**: 限制文本注意力范围（0=全局），降低显存占用\n"
                            "- **跨模态注意力**: 每 N 层插入视觉↔文本注意力（0=仅用投影层对齐）"
                        )
                        with gr.Row():
                            mm_visual_budget = gr.Slider(
                                0, 576, value=256, step=32,
                                label="视觉 Token 预算",
                                info="0=保留所有 patch tokens（~196-576个）",
                            )
                            mm_attn_window = gr.Slider(
                                0, 4096, value=0, step=256,
                                label="滑动窗口大小（tokens）",
                                info="0=全局注意力（推荐）",
                            )
                            mm_cross_attn_n = gr.Slider(
                                0, 8, value=4, step=1,
                                label="跨模态注意力间隔",
                                info="每 N 层插入一次（0=不插入）",
                            )

                    # ── 工具链集成 ──
                    with gr.Accordion("🔧 工具链集成（训练 Function Calling 能力）", open=False):
                        gr.Markdown(
                            "让多模态模型学会调用外部工具（API/函数），实现 **看图→调API→返回结果** 的闭环。\n"
                            "- 数据集需包含 `tool_call` / `tool_result` 类型的消息\n"
                            "- 格式示例: `{\"type\": \"tool_call\", \"content\": \"{\\\"name\\\": \\\"search\\\", ...}\"}`"
                        )
                        mm_tool_use = gr.Checkbox(label="启用工具调用训练", value=False)
                        with gr.Row():
                            mm_tool_format = gr.Dropdown(
                                label="工具调用格式",
                                choices=["chatml — ChatML 格式（推荐）",
                                         "react — ReAct 思考-行动格式",
                                         "json — 纯 JSON 格式"],
                                value="chatml — ChatML 格式（推荐）",
                            )
                            mm_tool_defs = gr.Textbox(
                                label="工具定义（JSON，可选）",
                                placeholder='[{"name": "web_search", "params": {"query": "str"}}]',
                                lines=2,
                            )

                    mm_vram_est = gr.Markdown("")
                    def _mm_vram_estimate(vis, aud, vis_enc, aud_enc, base_m):
                        try:
                            from core.multimodal import MultimodalConfig, VISION_ENCODERS, AUDIO_ENCODERS
                            v_key = (vis_enc or "").split(" — ")[0].strip() if vis_enc else "siglip-base"
                            a_key = (aud_enc or "").split(" — ")[0].strip() if aud_enc else "whisper-base"
                            cfg = MultimodalConfig(
                                enable_vision=bool(vis), enable_audio=bool(aud),
                                vision_encoder=v_key, audio_encoder=a_key,
                            )
                            est = cfg.estimate_vram_mb()
                            return f"📊 预估显存: **~{est}MB** ({'⚠️ 可能超 8GB' if est > 7500 else '✅ 8GB 内'})"
                        except Exception:
                            return ""
                    for _inp in [mm_vision, mm_audio, mm_vision_enc, mm_audio_enc, mm_base_model]:
                        _inp.change(_mm_vram_estimate,
                                    [mm_vision, mm_audio, mm_vision_enc, mm_audio_enc, mm_base_model],
                                    mm_vram_est)

                with gr.Row():
                    btn_mm_train = gr.Button("🔮 开始多模态训练", variant="primary", size="lg")
                mm_task_id = gr.Textbox(visible=False)
                mm_msg = gr.Textbox(label="状态", interactive=False)

                def _mm_train_submit(base_m, output, vis, aud, vis_enc, aud_enc,
                                     ds, proj, two_stage, lr_val, lr_llm, rank_val,
                                     epochs_val, batch_val, max_seq_val, ga_val,
                                     visual_budget, attn_window, cross_attn_n,
                                     tool_use, tool_fmt, tool_defs):
                    if not base_m:
                        return "", "❌ 请选择基座文本模型"
                    if not ds:
                        return "", "❌ 请选择多模态数据集"
                    try:
                        model_p = _resolve_model_path(base_m)
                        from core.multimodal import MultimodalConfig, MultimodalEngine
                        from core.task_queue import task_queue

                        v_key = (vis_enc or "").split(" — ")[0].strip()
                        a_key = (aud_enc or "").split(" — ")[0].strip()
                        t_fmt = (tool_fmt or "").split(" — ")[0].strip() or "chatml"
                        cfg = MultimodalConfig(
                            enable_vision=bool(vis), enable_audio=bool(aud),
                            vision_encoder=v_key, audio_encoder=a_key,
                            base_model=model_p,
                            projector_type=str(proj or "mlp2x"),
                            two_stage=bool(two_stage),
                            lr=float(lr_val or 1e-3),
                            lr_llm=float(lr_llm or 2e-5),
                            rank=int(rank_val or 32),
                            epochs=float(epochs_val or 3),
                            batch_size=int(batch_val or 1),
                            max_seq_len=int(max_seq_val or 2048),
                            gradient_accumulation_steps=int(ga_val or 8),
                            output_name=str(output or "multimodal_model"),
                            # 注意力窗口
                            visual_token_budget=int(visual_budget or 256),
                            attention_window=int(attn_window or 0),
                            cross_attn_every_n=int(cross_attn_n or 4),
                            # 工具链
                            enable_tool_use=bool(tool_use),
                            tool_format=t_fmt,
                            tool_definitions=str(tool_defs or ""),
                        )
                        engine = MultimodalEngine()
                        ds_path = str(DATASETS_DIR / ds)

                        def _run(task):
                            return engine.train(cfg, ds_path, task=task)

                        tid = task_queue.submit(f"🔮 多模态: {output or 'mm'}", _run)
                        modalities = []
                        if vis: modalities.append(f"👁️{v_key}")
                        if aud: modalities.append(f"🔊{a_key}")
                        if tool_use: modalities.append("🔧工具")
                        return tid, f"✅ 已提交 | {' + '.join(modalities)} | 任务: {tid}"
                    except Exception as e:
                        return "", f"❌ {e}"

                btn_mm_train.click(
                    _mm_train_submit,
                    [mm_base_model, mm_output, mm_vision, mm_audio, mm_vision_enc, mm_audio_enc,
                     mm_dataset, mm_projector, mm_two_stage, mm_lr, mm_lr_llm, mm_rank,
                     mm_epochs, mm_batch, mm_max_seq, mm_ga,
                     mm_visual_budget, mm_attn_window, mm_cross_attn_n,
                     mm_tool_use, mm_tool_format, mm_tool_defs],
                    [mm_task_id, mm_msg],
                )


            # ──────────────────────────────────────────────────
            # 模式 G: 🏗️ 从零预训练 / 继续预训练
            # ──────────────────────────────────────────────────
            with gr.Accordion("🏗️ 从零构建 & 继续预训练 — 自定义架构，全参数训练", open=False):
                gr.Markdown(
                    "两种模式:\n"
                    "- **从零构建**: 选架构 → 训 Tokenizer → 随机权重预训练 → 全新模型\n"
                    "- **继续预训练 (CPT)**: 在现有模型上灌入新领域语料，全参数训练\n\n"
                    "> 💡 8GB 显存可训 ≤ 300M 参数，500M+ 自动 CPU offload"
                )

                pt_mode = gr.Radio(
                    ["🆕 从零构建", "📚 继续预训练 (CPT)"],
                    value="🆕 从零构建", label="模式",
                )

                # ── 从零构建面板 ──
                pt_scratch_col = gr.Column(visible=True)
                with pt_scratch_col:
                    with gr.Row():
                        pt_preset = gr.Dropdown(
                            label="架构预设",
                            choices=[
                                "nano-25M — Nano (~25M) 学习实验",
                                "micro-40M — Micro (~40M) 轻量任务",
                                "mini-100M — Mini (~100M) GPT-2 级",
                                "small-300M — Small (~300M) GPT-2 Medium",
                                "medium-500M — Medium (~500M) CPU offload",
                                "large-1B — Large (~1B) 需耐心",
                                "custom — 自定义架构",
                            ],
                            value="mini-100M — Mini (~100M) GPT-2 级", scale=2,
                        )
                        pt_vocab = gr.Slider(8000, 64000, value=32000, step=1000,
                                              label="词表大小", scale=1)
                    with gr.Accordion("🔧 自定义架构（仅 custom 时生效）", open=False):
                        with gr.Row():
                            pt_hidden = gr.Number(label="Hidden Size", value=768, precision=0)
                            pt_layers = gr.Number(label="Layers", value=12, precision=0)
                            pt_heads = gr.Number(label="Attn Heads", value=12, precision=0)
                        with gr.Row():
                            pt_kv_heads = gr.Number(label="KV Heads (GQA)", value=4, precision=0)
                            pt_inter = gr.Number(label="MLP 中间层", value=2048, precision=0)
                            pt_max_pos = gr.Number(label="最大位置", value=2048, precision=0)

                # ── 继续预训练面板 ──
                pt_cpt_col = gr.Column(visible=False)
                with pt_cpt_col:
                    with gr.Row():
                        pt_cpt_model = gr.Dropdown(
                            label="基座模型", choices=_list_all_model_dirs(),
                            scale=2, info="要继续预训练的已有模型",
                        )
                        pt_cpt_hf = gr.Textbox(
                            label="或 HuggingFace ID", placeholder="Qwen/Qwen2.5-0.5B", scale=1,
                        )

                def _on_pt_mode(mode):
                    is_scratch = "从零" in (mode or "")
                    return gr.update(visible=is_scratch), gr.update(visible=not is_scratch)
                pt_mode.change(_on_pt_mode, pt_mode, [pt_scratch_col, pt_cpt_col])

                # ── 共享参数 ──
                pt_corpus = gr.Dropdown(label="训练语料", choices=_list_datasets(),
                                         multiselect=True, info="纯文本文件或 JSONL")
                with gr.Row():
                    pt_output = gr.Textbox(label="输出名称", value="my_pretrain", scale=1)
                    pt_lr = gr.Number(label="学习率", value=3e-4, precision=6, scale=1)
                    pt_batch = gr.Number(label="Batch Size", value=1, precision=0, scale=1)
                with gr.Row():
                    pt_epochs = gr.Number(label="Epochs", value=1, precision=1)
                    pt_max_seq = gr.Slider(128, 4096, value=512, step=128, label="Max Seq Len")
                    pt_ga = gr.Number(label="梯度累积", value=8, precision=0)
                    pt_max_steps = gr.Number(label="最大步数 (0=不限)", value=0, precision=0)

                with gr.Row():
                    btn_pretrain = gr.Button("🏗️ 开始训练", variant="primary", size="lg")
                pt_task_id = gr.Textbox(visible=False)
                pt_msg = gr.Textbox(label="状态", interactive=False)

                def _pretrain_dispatch(mode, preset_label, corpus, output,
                                       vocab, lr_val, batch_val, epochs_val,
                                       max_seq_val, ga_val, max_steps_val,
                                       custom_h, custom_l, custom_hd,
                                       custom_kv, custom_i, custom_mp,
                                       cpt_model, cpt_hf):
                    try:
                        if "从零" in (mode or ""):
                            preset_key = (preset_label or "").split(" — ")[0].strip()
                            return pretrain_submit(
                                preset=preset_key,
                                corpus_files=corpus,
                                output_name=str(output or ""),
                                vocab_size=int(vocab or 32000),
                                lr=float(lr_val or 3e-4),
                                batch_size=int(batch_val or 1),
                                epochs=float(epochs_val or 1),
                                max_seq_len=int(max_seq_val or 512),
                                ga_steps=int(ga_val or 8),
                                max_steps=int(max_steps_val or 0),
                                custom_hidden=int(custom_h or 768),
                                custom_layers=int(custom_l or 12),
                                custom_heads=int(custom_hd or 12),
                                custom_kv_heads=int(custom_kv or 4),
                                custom_intermediate=int(custom_i or 2048),
                                custom_max_pos=int(custom_mp or 2048),
                            )
                        else:
                            base_mode = "本机模型" if cpt_model else "HuggingFace 模型 ID"
                            return cpt_submit(
                                base_model_mode=base_mode,
                                base_model_hf=str(cpt_hf or ""),
                                base_model_local=str(cpt_model or ""),
                                corpus_files=corpus,
                                output_name=str(output or ""),
                                lr=float(lr_val or 1e-4),
                                batch_size=int(batch_val or 1),
                                epochs=float(epochs_val or 1),
                                max_seq_len=int(max_seq_val or 512),
                                ga_steps=int(ga_val or 8),
                                max_steps=int(max_steps_val or 0),
                            )
                    except Exception as e:
                        return "", f"❌ {e}"

                btn_pretrain.click(
                    _pretrain_dispatch,
                    [pt_mode, pt_preset, pt_corpus, pt_output,
                     pt_vocab, pt_lr, pt_batch, pt_epochs,
                     pt_max_seq, pt_ga, pt_max_steps,
                     pt_hidden, pt_layers, pt_heads,
                     pt_kv_heads, pt_inter, pt_max_pos,
                     pt_cpt_model, pt_cpt_hf],
                    [pt_task_id, pt_msg],
                )


            # ════════════════════════════════════════════════════════════════
            #  🧬 智能杂交 — 分析能力 + 取长补短
            # ════════════════════════════════════════════════════════════════
            with gr.Accordion("🧬 智能杂交 — 多模型取长补短合成新模型", open=False):
                gr.Markdown(
                    "自动分析每个模型的**能力特征**（多模态/推理/语言/代码/MoE），"
                    "用智能策略合并优点、去除缺点，产出兼具多种能力的新模型。\n\n"
                    "**与部署 Tab 的区别**: 这里做能力级别的智能分析+合并；部署 Tab 做张量级原始合并。\n\n"
                    "**原理**: 对每个模型计算 task vector（模型 - 基座），按能力维度加权后叠加。"
                )

                with gr.Row():
                    xb_base = gr.Dropdown(
                        label="🏗️ 基座模型（共同祖先）",
                        choices=_list_all_model_dirs(), scale=2,
                        info="所有模型应共享同一基座架构（如都基于 Llama-3.1-8B）",
                    )
                    xb_base_path = gr.Textbox(
                        label="或输入路径/HF ID", scale=2,
                        placeholder="meta-llama/Llama-3.1-8B",
                    )

                gr.Markdown("**选择参与杂交的模型**（至少 2 个，每个模型贡献不同能力）")
                with gr.Row():
                    xb_model_a = gr.Dropdown(label="模型 A", choices=_list_all_model_dirs(), scale=1)
                    xb_desc_a = gr.Textbox(label="能力标签", value="通用", scale=1,
                                            placeholder="如: 多模态, 代码, 推理, 中文")
                with gr.Row():
                    xb_model_b = gr.Dropdown(label="模型 B", choices=_list_all_model_dirs(), scale=1)
                    xb_desc_b = gr.Textbox(label="能力标签", value="通用", scale=1,
                                            placeholder="如: 多模态, 代码, 推理, 中文")
                with gr.Row():
                    xb_model_c = gr.Dropdown(label="模型 C（可选）", choices=_list_all_model_dirs(), scale=1)
                    xb_desc_c = gr.Textbox(label="能力标签", value="", scale=1,
                                            placeholder="留空 = 不参与")

                xb_analysis = gr.Markdown("_选择模型后自动分析能力差异_")

                with gr.Row():
                    xb_method = gr.Dropdown(
                        label="合并策略",
                        choices=["dare_ties — 随机丢弃+冲突消解（推荐）",
                                 "ties — 修剪冗余+符号投票",
                                 "task_arithmetic — 任务向量叠加",
                                 "dare_linear — 随机丢弃+线性平均"],
                        value="dare_ties — 随机丢弃+冲突消解（推荐）", scale=2,
                    )
                    xb_density = gr.Slider(0.1, 1.0, value=0.5, step=0.05,
                        label="密度", info="保留多少参数变化 (越低=越激进)", scale=1)
                    xb_weight = gr.Slider(0.1, 2.0, value=1.0, step=0.1,
                        label="能力权重", info="叠加强度 (>1 = 更强)", scale=1)

                with gr.Row():
                    xb_output = gr.Textbox(label="输出名", value="hybrid_smart", scale=2)
                    btn_xb_analyze = gr.Button("🔍 分析能力差异", scale=1)
                    btn_xb_go = gr.Button("🧬 开始智能杂交", variant="primary", scale=1)

                xb_task_id = gr.Textbox(visible=False)
                xb_msg = gr.Markdown("")

                # ── 能力分析 ──
                def _xb_analyze(base, base_path, ma, mb, mc):
                    """分析每个模型的能力特征"""
                    effective_base = (base_path or "").strip() or base
                    if not effective_base:
                        return "❌ 请先选择基座模型"
                    models_to_check = [(ma, "A"), (mb, "B")]
                    if mc:
                        models_to_check.append((mc, "C"))
                    if len([m for m, _ in models_to_check if m]) < 2:
                        return "❌ 至少选择 2 个模型"

                    lines = ["### 🔍 模型能力分析\n"]
                    for model_name, label in models_to_check:
                        if not model_name:
                            continue
                        try:
                            mp = _resolve_model_path(model_name)
                            abilities = _detect_model_abilities(mp)
                            ab_str = ", ".join(f"**{a}**" for a in abilities) if abilities else "通用文本"
                            lines.append(f"**模型 {label}** ({model_name}): {ab_str}")
                        except Exception as e:
                            lines.append(f"**模型 {label}** ({model_name}): ⚠️ {e}")

                    lines.append("\n---\n**合并策略建议**: ")
                    lines.append("使用 **DARE-TIES** 可以在保留各模型独特能力的同时消解参数冲突。")
                    lines.append("密度 0.3-0.5 适合差异大的模型，0.6-0.8 适合同质模型。")
                    return "\n".join(lines)

                btn_xb_analyze.click(
                    _xb_analyze,
                    [xb_base, xb_base_path, xb_model_a, xb_model_b, xb_model_c],
                    [xb_analysis],
                )

                # ── 提交杂交 ──
                def _xb_submit(base, base_path, ma, desc_a, mb, desc_b, mc, desc_c,
                               method_label, density, weight, output):
                    effective_base = (base_path or "").strip() or base
                    if not effective_base:
                        return "", "❌ 请选择基座模型"

                    # 收集参与模型
                    parts = []
                    for m, d in [(ma, desc_a), (mb, desc_b), (mc, desc_c)]:
                        if m:
                            parts.append((m, d or "通用"))
                    if len(parts) < 2:
                        return "", "❌ 至少选择 2 个模型"

                    method = (method_label or "").split(" — ")[0].strip()
                    model_names = [p[0] for p in parts]

                    # 解析基座路径
                    from pathlib import Path as _P
                    if _P(effective_base).is_dir():
                        base_resolved = effective_base
                    else:
                        try:
                            base_resolved = _resolve_model_path(effective_base)
                        except Exception:
                            base_resolved = effective_base  # 可能是 HF ID

                    return native_hybrid_submit(
                        base_resolved, method, model_names,
                        float(weight), float(density),
                        output or "hybrid_smart", "",
                    )

                btn_xb_go.click(
                    _xb_submit,
                    [xb_base, xb_base_path, xb_model_a, xb_desc_a,
                     xb_model_b, xb_desc_b, xb_model_c, xb_desc_c,
                     xb_method, xb_density, xb_weight, xb_output],
                    [xb_task_id, xb_msg],
                )

            def _on_train_tab_select():
                _invalidate_ds_cache()
                _invalidate_loras_cache()
                ds = _list_datasets()
                models = _list_all_model_dirs()
                mu = gr.update(choices=models)
                return (gr.update(choices=ds), gr.update(choices=ds),
                        gr.update(choices=ds), gr.update(choices=ds),
                        mu, mu, mu, mu, mu, mu,
                        mu, mu, mu, mu)
            tab_train.select(
                _on_train_tab_select, None,
                [dataset, api_dataset, mm_dataset, pt_corpus,
                 local_model_picker, api_student_picker,
                 doc_model_picker, evo_model_picker, mm_base_model,
                 pt_cpt_model,
                 xb_base, xb_model_a, xb_model_b, xb_model_c],
            )
        # ================================================================
        #  Tab 3: ✏️ 模型编辑
        # ================================================================
        with gr.Tab("✏️ 模型编辑") as tab_forge:
            # ── 全局共享: 当前编辑模型 (跨 Tab 联动核心) ──
            _active_edit_model = gr.State(value="")

            gr.Markdown(
                "### ✏️ 模型编辑工作台\n"
                "**完整工作流**: 选择模型 → 设定身份/知识库/安全规则 → 调参 → 一键烘焙导出\n\n"
                "> 💡 选择模型后所有面板自动同步加载，编辑实时保存。"
            )

            from core.forge_engine import ForgeEngine
            _forge = ForgeEngine()

            # ╔══════════════════════════════════════════╗
            #   模型编辑器
            # ╚══════════════════════════════════════════╝
            with gr.Accordion("✏️ 模型编辑器 — 人设 / 知识库 / 推理参数 / Modelfile", open=True):
                from core.model_editor import (
                    load_profile, save_profile, add_knowledge_doc,
                    remove_knowledge_doc, get_model_summary,
                    load_chat_template, save_chat_template,
                    generate_modelfile, generate_system_prompt_with_knowledge,
                )

                # ── 模型选择 + 刷新 ──
                with gr.Row():
                    me_model = gr.Dropdown(
                        label="🎯 选择要编辑的模型", choices=_list_all_model_dirs(),
                        scale=3, info="选择后所有面板自动加载，可在下方 Tab 中编辑",
                    )
                    btn_me_refresh_models = gr.Button("🔄", scale=0, min_width=50)

                # ── 模型信息卡片 — 选择后立即显示 ──
                me_model_card = gr.Markdown(
                    value=(
                        "---\n"
                        "#### 👆 请从上方下拉菜单选择一个模型\n\n"
                        "选择后将自动显示模型架构、编辑进度、显存估算等信息。\n\n"
                        "**没有模型?** 先去 🔥 训练 Tab 训练一个，或 🚀 部署 Tab 合并 LoRA。"
                    ),
                    elem_id="me-model-card",
                )

                # ── 快捷操作栏 ──
                with gr.Row(visible=False) as me_quick_actions:
                    btn_me_goto_eval = gr.Button("🧪 去评测/聊天测试", size="sm", scale=1)
                    btn_me_goto_deploy = gr.Button("🚀 去部署/导出", size="sm", scale=1)
                    btn_me_goto_train = gr.Button("🔥 用此模型训练", size="sm", scale=1)
                    with gr.Column(scale=2):
                        me_ready_badge = gr.Markdown("")

                # ── 详细摘要（折叠） ──
                me_summary = gr.Markdown("", elem_id="me-summary", visible=False)

                with gr.Tabs() as me_tabs:

                    # ── 身份 & 系统提示词 & 安全 ──
                    with gr.Tab("🎭 身份 & 提示词"):
                        with gr.Row():
                            me_name = gr.Textbox(label="模型名称", placeholder="如: 法律助手 v1.0")
                            me_author = gr.Textbox(label="作者", placeholder="如: ForgeX")
                        me_desc = gr.Textbox(label="模型描述", placeholder="这个模型擅长...",
                                              lines=2, info="简要说明模型用途")
                        with gr.Row():
                            me_tags = gr.Textbox(label="标签（逗号分隔）",
                                                  placeholder="法律,中文,问答", scale=2)
                            me_usecase = gr.Textbox(label="使用场景",
                                                     placeholder="合同审核、法规问答", scale=2)

                        gr.Markdown("---\n**系统提示词** — 定义模型的人设和行为准则")
                        me_system_prompt = gr.Textbox(
                            label="System Prompt",
                            placeholder="你是一位专业的法律顾问...",
                            lines=8,
                        )
                        with gr.Row():
                            btn_me_save_identity = gr.Button("💾 保存身份 & 提示词", variant="primary", scale=2)
                            btn_me_preview_sp = gr.Button("👀 预览完整提示词", scale=1)
                        me_identity_msg = gr.Markdown("")
                        me_sp_preview = gr.Textbox(label="完整系统提示词（含知识库+安全规则）", visible=False,
                                                    lines=6, interactive=False)

                        gr.Markdown("---\n**🛡️ 安全护栏**（可选）")
                        me_safety_on = gr.Checkbox(label="启用安全护栏", value=False)
                        with gr.Row():
                            me_refusal = gr.Textbox(
                                label="拒绝主题（每行一个）",
                                placeholder="政治敏感\n违法犯罪\n色情内容",
                                lines=3, scale=1,
                            )
                            me_content_filter = gr.Textbox(
                                label="自定义安全规则",
                                placeholder="不要透露训练数据来源，回答要客观中立",
                                lines=3, scale=1,
                            )
                        with gr.Row():
                            me_disclaimer_on = gr.Checkbox(label="附加免责声明", value=False)
                            me_disclaimer = gr.Textbox(
                                label="免责声明文本",
                                placeholder="以上内容仅供参考。",
                                scale=2,
                            )
                        btn_me_save_safety = gr.Button("💾 保存安全设置", variant="secondary")
                        me_safety_msg = gr.Markdown("")

                    # ── 知识库 ──
                    with gr.Tab("📚 知识库"):
                        gr.Markdown(
                            "上传参考文档，模型回答时会自动检索相关内容。\n"
                            "支持: txt, md, json, csv, pdf\n\n"
                            "文档会被分块索引，嵌入系统提示词中（RAG 模式）。"
                        )
                        me_kb_status = gr.Markdown("_选择模型后自动显示知识库状态_")
                        with gr.Row():
                            me_kb_file = gr.File(label="上传文档", file_types=[".txt", ".md", ".json", ".csv", ".pdf"])
                            me_kb_name = gr.Textbox(label="文档名（可选）", placeholder="自动使用文件名")
                        with gr.Row():
                            btn_me_kb_add = gr.Button("📤 添加到知识库", variant="primary")
                            btn_me_kb_refresh = gr.Button("🔄 刷新列表")
                        me_kb_list = gr.Dataframe(
                            headers=["文档名", "分块数", "字符数", "添加时间"],
                            datatype=["str", "number", "number", "str"],
                            row_count=5, col_count=4, interactive=False,
                        )
                        with gr.Row():
                            me_kb_del_name = gr.Textbox(label="要删除的文档名", scale=2)
                            btn_me_kb_del = gr.Button("🗑️ 删除", variant="stop", scale=1)
                        me_kb_msg = gr.Markdown("")

                    # ── 推理参数 ──
                    with gr.Tab("⚙️ 推理参数"):
                        gr.Markdown("设置模型默认推理参数。这些值会保存到 Modelfile 和模型档案中。")
                        with gr.Row():
                            me_temp = gr.Slider(0.0, 2.0, value=0.7, step=0.05,
                                label="Temperature", info="创造性 (0=确定, 1=平衡, 2=狂野)")
                            me_top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05,
                                label="Top-P", info="核采样阈值")
                        with gr.Row():
                            me_top_k = gr.Number(label="Top-K", value=50, precision=0,
                                info="候选词数量 (0=不限)")
                            me_repeat_penalty = gr.Slider(1.0, 2.0, value=1.1, step=0.05,
                                label="重复惩罚", info="防止重复 (1.0=不惩罚)")
                        with gr.Row():
                            me_max_tokens = gr.Number(label="最大 Token 数", value=2048, precision=0,
                                info="单次回复最大长度")
                            me_stop_seq = gr.Textbox(label="停止序列（逗号分隔）",
                                placeholder="</s>,<|im_end|>,User:",
                                info="遇到这些字符串停止生成")
                        btn_me_save_params = gr.Button("💾 保存推理参数", variant="primary")
                        me_params_msg = gr.Markdown("")

                    # ── 烘焙导出 ──
                    with gr.Tab("🔥 烘焙导出"):
                        me_bake_checklist = gr.Markdown("_选择模型后显示烘焙清单_")
                        gr.Markdown(
                            "---\n"
                            "**烘焙** 会将系统提示词、知识库、安全规则、推理参数"
                            "全部写入模型文件。\n"
                            "烘焙后模型是**开箱即用**的 — 任何工具加载都自动带上这些设定。\n\n"
                            "**完整流水线**: 烘焙 → 合并 LoRA → 导出 GGUF → 生成 Modelfile → 导入 Ollama"
                        )
                        with gr.Row():
                            btn_me_bake = gr.Button("🔥 仅烘焙（写入模型文件）", variant="secondary", scale=1)
                            me_bake_msg = gr.Markdown("")
                        gr.Markdown("---\n**一键烘焙导出** — 全自动流水线")
                        with gr.Row():
                            me_export_name = gr.Textbox(label="导出名称", value="", placeholder="自动使用模型名",
                                scale=2)
                            me_export_quant = gr.Dropdown(label="GGUF 量化",
                                choices=["F16", "Q8_0", "Q4_K_M", "Q5_K_M", "Q6_K"],
                                value="Q4_K_M", scale=1)
                        me_export_ollama = gr.Checkbox(label="自动导入 Ollama", value=False,
                            info="导出完成后自动执行 ollama create")
                        btn_me_bake_export = gr.Button("🚀 一键烘焙导出 (Bake → GGUF → Modelfile)", variant="primary")
                        me_bake_export_tid = gr.Textbox(visible=False)
                        me_bake_export_msg = gr.Textbox(label="导出状态", interactive=False, lines=2)

                # ════════════════════════════════════════════════════
                #  核心联动逻辑（增强版 — 模型卡片 + 跨 Tab 同步）
                # ════════════════════════════════════════════════════

                # ---- 通用：生成烘焙清单 ----
                def _build_bake_checklist(profile):
                    """根据 profile 生成烘焙前的检查清单"""
                    items = []
                    sp = profile.get("system_prompt", "")
                    items.append(f"{'✅' if sp else '⬜'} 系统提示词 {'(' + str(len(sp)) + '字)' if sp else '— 未设置'}")
                    kb = profile.get("knowledge_docs", [])
                    items.append(f"{'✅' if kb else '⬜'} 知识库 {'(' + str(len(kb)) + '个文档)' if kb else '— 空'}")
                    safety = profile.get("safety", {})
                    items.append(f"{'✅' if safety.get('enabled') else '⬜'} 安全护栏 {'— 启用' if safety.get('enabled') else '— 未启用'}")
                    params = profile.get("parameters", {})
                    items.append(f"{'✅' if params else '⬜'} 推理参数 {'— 已配置' if params else '— 使用默认'}")
                    n_ready = sum(1 for i in items if i.startswith('✅'))
                    header = f"### 📋 烘焙清单 ({n_ready}/4 项已配置)"
                    return header + "\n\n" + "\n".join(items)

                # ---- 模型信息卡片（选择后立即显示）----
                def _build_model_card(model_name, profile, model_path):
                    """生成富信息模型卡片"""
                    d = Path(model_path)
                    clean = d.name
                    arch = profile.get("_architecture", {})
                    params_est = profile.get("_estimated_params", "未知")

                    # 类型徽章
                    has_lora = (d / "adapter_config.json").exists()
                    has_weights = any(d.glob("model*.safetensors")) or (d / "pytorch_model.bin").exists()
                    if has_lora:
                        type_badge = "🔌 **LoRA 适配器**"
                    elif has_weights:
                        type_badge = "📦 **完整模型**"
                    else:
                        type_badge = "📄 **模型配置**"

                    lines = ["---", f"#### {type_badge}  `{clean}`"]

                    # 架构表格
                    if arch:
                        mtype = arch.get("model_type", "?")
                        layers = arch.get("num_hidden_layers", "?")
                        hidden = arch.get("hidden_size", "?")
                        heads = arch.get("num_attention_heads", "?")
                        inter = arch.get("intermediate_size", "?")
                        vocab = arch.get("vocab_size", "?")
                        lines.append(
                            f"\n| 架构 | 层数 | Hidden | 注意力头 | MLP | 词表 | 参数量 |"
                        )
                        lines.append(f"|:--|:--|:--|:--|:--|:--|:--|")
                        lines.append(
                            f"| {mtype} | {layers} | {hidden} | {heads} | {inter} | {vocab} | ~{params_est} |"
                        )
                        # VRAM 估算
                        try:
                            p = float(str(params_est).replace("B", ""))
                            fit8g = "✅ 8GB 显卡可跑 Q4" if p * 0.6 < 7 else "⚠️ 8GB 显卡可能不够"
                            lines.append(
                                f"\n💾 显存估算: FP16≈{p*2:.1f}GB  Q4≈{p*0.6:.1f}GB — {fit8g}"
                            )
                        except Exception:
                            pass

                    # 磁盘大小
                    try:
                        model_files = [f for f in d.iterdir() if f.suffix in (".safetensors", ".bin", ".gguf")]
                        total_mb = sum(f.stat().st_size for f in model_files) / 1e6
                        lines.append(f"\n📁 磁盘: {len(model_files)} 个权重文件 | {total_mb:,.0f} MB")
                    except Exception:
                        pass

                    # 编辑进度条
                    sp = profile.get("system_prompt", "")
                    kb = profile.get("knowledge_docs", [])
                    safety = profile.get("safety", {})
                    user_params = profile.get("parameters", {})
                    checks = [bool(sp), bool(kb), safety.get("enabled", False), bool(user_params)]
                    n_done = sum(checks)
                    bar = "🟩" * n_done + "⬜" * (4 - n_done)
                    parts = []
                    if sp: parts.append(f"提示词({len(sp)}字)")
                    if kb: parts.append(f"知识库({len(kb)}篇)")
                    if safety.get("enabled"): parts.append("安全护栏")
                    if user_params: parts.append("推理参数")
                    status_text = " · ".join(parts) if parts else "尚未编辑 — 在下方 Tab 中开始"
                    lines.append(f"\n**编辑进度** {bar} {n_done}/4 — {status_text}")

                    return "\n".join(lines)

                def _build_ready_badge(profile):
                    """生成就绪状态徽章"""
                    sp = profile.get("system_prompt", "")
                    kb = profile.get("knowledge_docs", [])
                    safety = profile.get("safety", {})
                    user_params = profile.get("parameters", {})
                    checks = [bool(sp), bool(kb), safety.get("enabled", False), bool(user_params)]
                    n = sum(checks)
                    if n == 0:
                        return "⬜ 空白模型 — 请先设定身份和系统提示词"
                    elif n <= 2:
                        return f"🟡 基础配置 ({n}/4) — 建议补充更多设定"
                    elif n == 3:
                        return f"🟢 接近就绪 ({n}/4) — 再配一项即可烘焙"
                    else:
                        return f"✅ 全部就绪 (4/4) — 可以烘焙导出了！"

                # ---- 核心：加载模型到所有面板 ----
                def _me_load_all(model_name):
                    """选择模型后自动加载全部信息到所有面板"""
                    _empty_arch = "_选择源模型后显示架构信息_"
                    empty_card = (
                        "---\n"
                        "#### 👆 请从上方选择一个模型\n\n"
                        "选择后将显示模型架构、编辑进度、显存估算等信息。\n\n"
                        "**没有模型?** 先去 🔥训练 Tab 训练，或 🚀部署 Tab 合并 LoRA。"
                    )
                    empty = (empty_card,
                             gr.update(visible=False), gr.update(visible=False, value=""),
                             "", "",
                             "", "", "", "", "", "", "",
                             0.7, 0.9, 50, 1.1, 2048, "",
                             [], "_选择模型后自动显示知识库状态_",
                             False, "", "", False, "",
                             "_选择模型后显示烘焙清单_", "",
                             gr.update(), _empty_arch)
                    if not model_name:
                        return empty
                    try:
                        clean = _extract_model_name_from_label(model_name)
                        mp = str(LORAS_DIR / clean)
                        profile = load_profile(mp)
                        summary = get_model_summary(mp)
                        model_card = _build_model_card(model_name, profile, mp)
                        ready_badge = _build_ready_badge(profile)
                        params = profile.get("parameters", {})
                        kb_docs = profile.get("knowledge_docs", [])
                        kb_rows = [[d["name"], d.get("chunks", 0), d.get("chars", 0),
                                     d.get("added_at", "")] for d in kb_docs]
                        total_chunks = sum(d.get("chunks", 0) for d in kb_docs)
                        total_chars = sum(d.get("chars", 0) for d in kb_docs)
                        if kb_docs:
                            kb_status = (f"### 📚 知识库: {len(kb_docs)} 个文档\n"
                                         f"共 {total_chunks} 个分块 | {total_chars:,} 字符")
                        else:
                            kb_status = "### 📚 知识库: 空\n上传文档后，模型回答会自动检索相关内容。"
                        safety = profile.get("safety", {})
                        bake_cl = _build_bake_checklist(profile)
                        export_name = clean
                        # 手术区：架构简报
                        try:
                            from core.model_editor import get_model_arch_brief
                            arch_brief = get_model_arch_brief(mp)
                        except Exception:
                            arch_brief = "_架构信息读取失败_"
                        return (
                            model_card,
                            gr.update(visible=True),   # me_quick_actions
                            gr.update(visible=True, value=summary),  # me_summary
                            ready_badge,                # me_ready_badge
                            model_name,                 # _active_edit_model
                            profile.get("name", clean),
                            profile.get("author", ""),
                            profile.get("description", ""),
                            ", ".join(profile.get("tags", [])),
                            profile.get("use_case", ""),
                            profile.get("system_prompt", ""),
                            "",  # me_sp_preview reset
                            params.get("temperature", 0.7),
                            params.get("top_p", 0.9),
                            params.get("top_k", 50),
                            params.get("repeat_penalty", 1.1),
                            params.get("max_tokens", 2048),
                            ", ".join(params.get("stop_sequences", [])),
                            kb_rows,
                            kb_status,
                            safety.get("enabled", False),
                            "\n".join(safety.get("refusal_topics", [])),
                            safety.get("content_filter", ""),
                            safety.get("require_disclaimer", False),
                            safety.get("disclaimer_text", ""),
                            bake_cl,
                            export_name,
                            gr.update(value=model_name),  # → exp_model 同步
                            arch_brief,                    # → exp_summary 同步
                        )
                    except Exception as e:
                        err_card = f"---\n#### ❌ 加载失败\n`{e}`\n\n请检查模型目录是否完整。"
                        return (err_card, gr.update(visible=False),
                                gr.update(visible=False, value=f"### ❌ 加载失败\n{e}"),
                                "", "",
                                ) + ("",) * 7 + (
                                0.7, 0.9, 50, 1.1, 2048, "", [], "_加载失败_",
                                False, "", "", False, "", f"_加载失败: {e}_", "",
                                gr.update(), f"_加载失败: {e}_")

                # (output list + model.change wiring moved after expansion section where exp_model is defined)

                # 刷新模型列表
                def _me_refresh_models():
                    _invalidate_loras_cache()
                    return gr.update(choices=_list_all_model_dirs())
                btn_me_refresh_models.click(_me_refresh_models, None, [me_model])

                # ---- 刷新摘要 + 烘焙清单（保存后联动更新）----
                def _me_refresh_summary(model_name):
                    """保存操作后刷新: 模型卡片 + 摘要 + 烘焙清单 + 就绪徽章"""
                    if not model_name:
                        return "---\n#### 👆 请先选择模型", "", "_选择模型后显示烘焙清单_", ""
                    try:
                        mp = _resolve_model_path(model_name)
                        profile = load_profile(mp)
                        summary = get_model_summary(mp)
                        card = _build_model_card(model_name, profile, mp)
                        bake_cl = _build_bake_checklist(profile)
                        badge = _build_ready_badge(profile)
                        return card, summary, bake_cl, badge
                    except Exception:
                        return "---\n#### ⚠️ 刷新失败", "", "_刷新失败_", ""

                # ---- 身份保存（保存后刷新摘要）----
                def _me_save_identity(model_name, name, author, desc, tags, usecase, sp):
                    if not model_name:
                        return "❌ 请先选择模型", gr.update(), gr.update(), gr.update(), gr.update()
                    try:
                        mp = _resolve_model_path(model_name)
                        profile = load_profile(mp)
                        profile["name"] = str(name or "")
                        profile["author"] = str(author or "")
                        profile["description"] = str(desc or "")
                        profile["tags"] = [t.strip() for t in str(tags or "").split(",") if t.strip()]
                        profile["use_case"] = str(usecase or "")
                        profile["system_prompt"] = str(sp or "")
                        save_profile(mp, profile)
                        card, summary, bake_cl, badge = _me_refresh_summary(model_name)
                        return (f"✅ 已保存身份 & 提示词 ({len(profile['system_prompt'])} 字)",
                                card, summary, bake_cl, badge)
                    except Exception as e:
                        return f"❌ 保存失败: {e}", gr.update(), gr.update(), gr.update(), gr.update()

                btn_me_save_identity.click(
                    _me_save_identity,
                    [me_model, me_name, me_author, me_desc, me_tags, me_usecase, me_system_prompt],
                    [me_identity_msg, me_model_card, me_summary, me_bake_checklist, me_ready_badge],
                )

                # ---- 预览完整提示词 ----
                def _me_preview_sp(model_name):
                    if not model_name:
                        return gr.update(visible=False, value="")
                    try:
                        mp = _resolve_model_path(model_name)
                        full_sp = generate_system_prompt_with_knowledge(mp)
                        return gr.update(visible=True, value=full_sp or "(空)")
                    except Exception as e:
                        return gr.update(visible=True, value=f"❌ {e}")

                btn_me_preview_sp.click(_me_preview_sp, [me_model], [me_sp_preview])

                # ---- 推理参数保存（保存后刷新摘要）----
                def _me_save_params(model_name, temp, top_p, top_k, rp, max_t, stop):
                    if not model_name:
                        return "❌ 请先选择模型", gr.update(), gr.update(), gr.update(), gr.update()
                    try:
                        mp = _resolve_model_path(model_name)
                        profile = load_profile(mp)
                        profile["parameters"] = {
                            "temperature": float(temp if temp is not None else 0.7),
                            "top_p": float(top_p if top_p is not None else 0.9),
                            "top_k": int(top_k if top_k is not None else 50),
                            "repeat_penalty": float(rp if rp is not None else 1.1),
                            "max_tokens": int(max_t if max_t is not None else 2048),
                            "stop_sequences": [s.strip() for s in str(stop or "").split(",") if s.strip()],
                        }
                        save_profile(mp, profile)
                        card, summary, bake_cl, badge = _me_refresh_summary(model_name)
                        return "✅ 推理参数已保存", card, summary, bake_cl, badge
                    except Exception as e:
                        return f"❌ {e}", gr.update(), gr.update(), gr.update(), gr.update()

                btn_me_save_params.click(
                    _me_save_params,
                    [me_model, me_temp, me_top_p, me_top_k, me_repeat_penalty, me_max_tokens, me_stop_seq],
                    [me_params_msg, me_model_card, me_summary, me_bake_checklist, me_ready_badge],
                )

                # ---- 知识库操作（操作后刷新摘要+状态+清单）----
                def _me_kb_add(model_name, file_obj, doc_name):
                    if not model_name or not file_obj:
                        return "❌ 请选择模型和文档", [], gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                    try:
                        mp = _resolve_model_path(model_name)
                        fpath = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
                        result = add_knowledge_doc(mp, fpath, doc_name or "")
                        profile = load_profile(mp)
                        kb_docs = profile.get("knowledge_docs", [])
                        kb_rows = [[d["name"], d.get("chunks", 0), d.get("chars", 0),
                                     d.get("added_at", "")] for d in kb_docs]
                        total_chunks = sum(d.get("chunks", 0) for d in kb_docs)
                        total_chars = sum(d.get("chars", 0) for d in kb_docs)
                        kb_status = (f"### 📚 知识库: {len(kb_docs)} 个文档\n"
                                     f"共 {total_chunks} 个分块 | {total_chars:,} 字符")
                        card, summary, bake_cl, badge = _me_refresh_summary(model_name)
                        return (f"✅ 已添加: {result['name']} ({result['chunks']} chunks)",
                                kb_rows, kb_status, card, summary, bake_cl, badge)
                    except Exception as e:
                        return f"❌ {e}", [], gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

                btn_me_kb_add.click(
                    _me_kb_add, [me_model, me_kb_file, me_kb_name],
                    [me_kb_msg, me_kb_list, me_kb_status, me_model_card, me_summary, me_bake_checklist, me_ready_badge],
                )

                def _me_kb_refresh(model_name):
                    if not model_name:
                        return [], "_选择模型后自动显示知识库状态_"
                    try:
                        mp = _resolve_model_path(model_name)
                        profile = load_profile(mp)
                        kb_docs = profile.get("knowledge_docs", [])
                        kb_rows = [[d["name"], d.get("chunks", 0), d.get("chars", 0),
                                     d.get("added_at", "")] for d in kb_docs]
                        total_chunks = sum(d.get("chunks", 0) for d in kb_docs)
                        total_chars = sum(d.get("chars", 0) for d in kb_docs)
                        if kb_docs:
                            kb_status = (f"### 📚 知识库: {len(kb_docs)} 个文档\n"
                                         f"共 {total_chunks} 个分块 | {total_chars:,} 字符")
                        else:
                            kb_status = "### 📚 知识库: 空"
                        return kb_rows, kb_status
                    except Exception:
                        return [], "_加载失败_"

                btn_me_kb_refresh.click(_me_kb_refresh, [me_model], [me_kb_list, me_kb_status])

                def _me_kb_del(model_name, doc_name):
                    if not model_name or not doc_name:
                        return "❌ 请选择模型和文档名", [], gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                    try:
                        mp = _resolve_model_path(model_name)
                        remove_knowledge_doc(mp, doc_name)
                        kb_rows, kb_status = _me_kb_refresh(model_name)
                        card, summary, bake_cl, badge = _me_refresh_summary(model_name)
                        return f"✅ 已删除: {doc_name}", kb_rows, kb_status, card, summary, bake_cl, badge
                    except Exception as e:
                        return f"❌ {e}", [], gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

                btn_me_kb_del.click(
                    _me_kb_del, [me_model, me_kb_del_name],
                    [me_kb_msg, me_kb_list, me_kb_status, me_model_card, me_summary, me_bake_checklist, me_ready_badge],
                )


                # ── 安全护栏保存（保存后刷新摘要）──
                def _me_save_safety(model_name, enabled, refusal, cf, disc_on, disc_text):
                    if not model_name:
                        return "❌ 请先选择模型", gr.update(), gr.update(), gr.update(), gr.update()
                    try:
                        mp = _resolve_model_path(model_name)
                        profile = load_profile(mp)
                        profile["safety"] = {
                            "enabled": bool(enabled),
                            "refusal_topics": [t.strip() for t in str(refusal or "").split("\n") if t.strip()],
                            "content_filter": str(cf or ""),
                            "require_disclaimer": bool(disc_on),
                            "disclaimer_text": str(disc_text or ""),
                        }
                        save_profile(mp, profile)
                        n_topics = len(profile["safety"]["refusal_topics"])
                        card, summary, bake_cl, badge = _me_refresh_summary(model_name)
                        return (f"✅ 安全设置已保存 | {'启用' if enabled else '停用'} | {n_topics} 个拒绝主题",
                                card, summary, bake_cl, badge)
                    except Exception as e:
                        return f"❌ {e}", gr.update(), gr.update(), gr.update(), gr.update()
                btn_me_save_safety.click(
                    _me_save_safety,
                    [me_model, me_safety_on, me_refusal, me_content_filter, me_disclaimer_on, me_disclaimer],
                    [me_safety_msg, me_model_card, me_summary, me_bake_checklist, me_ready_badge],
                )

                # ── 输出格式保存 ──

                # ── 角色预设 ──


                # ── 烘焙（烘焙后刷新摘要 + 卡片）──
                def _me_bake_only(model_name):
                    if not model_name:
                        return "❌ 请先选择模型", gr.update(), gr.update()
                    try:
                        mp = _resolve_model_path(model_name)
                        from core.model_editor import bake_profile_into_model
                        result = bake_profile_into_model(mp)
                        card, _s, _b, _bg = _me_refresh_summary(model_name)
                        return (
                            f"✅ **烘焙完成!** 所有编辑已写入 `{Path(result).name}`\n\n"
                            f"**下一步**: 点击上方 **🧪去评测** 测试效果，或 **一键烘焙导出** 生成 GGUF。",
                            card, gr.update()
                        )
                    except Exception as e:
                        return f"❌ 烘焙失败: {e}", gr.update(), gr.update()
                btn_me_bake.click(_me_bake_only, [me_model], [me_bake_msg, me_model_card, me_summary])

                # ── 一键烘焙导出 ──
                def _me_bake_export(model_name, export_name, quant, ollama_import):
                    if not model_name:
                        return "", "❌ 请先选择模型"
                    try:
                        mp = _resolve_model_path(model_name)
                        from core.model_editor import bake_and_export
                        def _run(task):
                            return bake_and_export(
                                mp,
                                output_name=export_name or "",
                                quant_type=quant or "Q4_K_M",
                                ollama_import=bool(ollama_import),
                                task=task,
                            )
                        name = f"🔥 烘焙导出: {_extract_model_name_from_label(model_name)} → {quant}"
                        tid = task_queue.submit(name, _run)
                        return tid, (
                            f"✅ 任务已提交: `{tid}`\n"
                            f"流水线: 烘焙 → GGUF({quant}) → Modelfile"
                            f"{' → Ollama 导入' if ollama_import else ''}\n"
                            f"📋 切换到 **任务中心** Tab 查看进度"
                        )
                    except Exception as e:
                        return "", f"❌ {e}"
                btn_me_bake_export.click(
                    _me_bake_export,
                    [me_model, me_export_name, me_export_quant, me_export_ollama],
                    [me_bake_export_tid, me_bake_export_msg],
                )

            # ── 知识膨胀 / 模型手术 ──
            with gr.Accordion("💪 模型手术 — 膨胀 / 嫁接 / 词表扩展 / MoE 升级", open=False):
                gr.Markdown(
                    "在**不训练**的情况下修改模型结构：\n"
                    "- **深度膨胀**: 复制中间层加深网络 → 参数↑ 容量↑\n"
                    "- **宽度膨胀**: Net2Net 扩展 hidden/MLP 维度 → 更宽网络\n"
                    "- **混合膨胀**: 同时加深+加宽\n"
                    "- **知识嫁接**: 从大模型移植层到小模型 → 继承知识\n"
                    "- **词表扩展**: 添加新 token (专业术语/新语言)\n"
                    "- **MoE 升级**: Dense → MoE 专家混合 → 容量暴增\n\n"
                    "> 膨胀后建议接 SFT 训练激活新参数"
                )
                with gr.Row():
                    exp_model = gr.Dropdown(label="🎯 源模型", choices=_list_all_model_dirs(), scale=2,
                                            info="选择后自动显示架构信息")
                    exp_output = gr.Textbox(label="输出名", value="expanded_model", scale=1)
                exp_summary = gr.Markdown("_选择源模型后显示架构信息_")

                exp_method = gr.Radio(
                    ["depth — 深度膨胀", "width — 宽度膨胀", "hybrid — 混合膨胀",
                     "graft — 知识嫁接", "vocab — 词表扩展", "moe — MoE 升级"],
                    value="depth — 深度膨胀", label="操作类型",
                )

                # 深度参数
                exp_depth_col = gr.Column(visible=True)
                with exp_depth_col:
                    with gr.Row():
                        exp_depth_layers = gr.Slider(1, 24, value=4, step=1, label="新增层数")
                        exp_depth_strat = gr.Dropdown(
                            ["repeat_middle", "repeat_all", "repeat_top", "interleave"],
                            value="repeat_middle", label="复制策略",
                        )
                        exp_noise = gr.Slider(0.0, 0.1, value=0.01, step=0.005, label="噪声系数")

                # 宽度参数
                exp_width_col = gr.Column(visible=False)
                with exp_width_col:
                    with gr.Row():
                        exp_target_h = gr.Number(label="目标 Hidden Size (0=不变)", value=0, precision=0)
                        exp_target_inter = gr.Number(label="目标 MLP 中间层 (0=自动)", value=0, precision=0)

                # 混合参数
                exp_hybrid_col = gr.Column(visible=False)
                with exp_hybrid_col:
                    with gr.Row():
                        exp_hyb_layers = gr.Number(label="目标层数 (0=不加深)", value=0, precision=0)
                        exp_hyb_hidden = gr.Number(label="目标 Hidden (0=不加宽)", value=0, precision=0)
                        exp_hyb_inter = gr.Number(label="目标 MLP (0=自动)", value=0, precision=0)

                # 嫁接参数
                exp_graft_col = gr.Column(visible=False)
                with exp_graft_col:
                    gr.Markdown("**知识源** — 支持本机模型下拉选择，也支持直接输入 HF 模型路径")
                    with gr.Row():
                        exp_large_model = gr.Dropdown(
                            label="大模型（本机列表）", choices=_list_all_model_dirs(),
                            info="选择后显示架构对比", scale=2,
                        )
                        exp_large_path = gr.Textbox(
                            label="或输入 HF 模型路径",
                            placeholder="如: /models/Llama-3.1-70B 或 meta-llama/Llama-3.1-70B",
                            scale=2, info="优先使用此路径（留空则用左边下拉）",
                        )
                    exp_graft_info = gr.Markdown("_选择大模型后显示架构对比_")
                    with gr.Row():
                        exp_graft_n = gr.Slider(1, 12, value=4, step=1, label="嫁接层数")

                # 词表扩展参数
                exp_vocab_col = gr.Column(visible=False)
                with exp_vocab_col:
                    with gr.Row():
                        exp_new_tokens = gr.Textbox(label="新增 Token（逗号分隔）",
                                                     placeholder="<think>, <tool>, 医学术语...", scale=2)
                        exp_vocab_size = gr.Number(label="目标词表大小 (0=仅加token)", value=0, precision=0)

                # MoE 参数
                exp_moe_col = gr.Column(visible=False)
                with exp_moe_col:
                    with gr.Row():
                        exp_moe_experts = gr.Slider(2, 8, value=4, step=1, label="专家数量")
                        exp_moe_topk = gr.Slider(1, 4, value=2, step=1, label="Top-K 路由")
                        exp_moe_layers_str = gr.Textbox(label="MoE 层 (空=全部)", placeholder="8,12,16", value="")

                def _on_exp_method(method):
                    key = (method or "").split(" — ")[0].strip()
                    return (gr.update(visible=key == "depth"),
                            gr.update(visible=key == "width"),
                            gr.update(visible=key == "hybrid"),
                            gr.update(visible=key == "graft"),
                            gr.update(visible=key == "vocab"),
                            gr.update(visible=key == "moe"))
                exp_method.change(_on_exp_method, exp_method,
                                   [exp_depth_col, exp_width_col, exp_hybrid_col,
                                    exp_graft_col, exp_vocab_col, exp_moe_col])

                with gr.Row():
                    btn_expand = gr.Button("💪 执行手术", variant="primary", size="lg")
                exp_task_id = gr.Textbox(visible=False)
                exp_msg = gr.Textbox(label="状态", interactive=False)

                # ── 手术区：源模型选择联动 ──
                def _on_exp_model_change(model_name):
                    """手术区选择源模型 → 显示架构信息"""
                    if not model_name:
                        return "_选择源模型后显示架构信息_"
                    try:
                        from core.model_editor import get_model_arch_brief
                        mp = _resolve_model_path(model_name)
                        return get_model_arch_brief(mp)
                    except Exception as e:
                        return f"⚠️ {e}"
                exp_model.change(_on_exp_model_change, [exp_model], [exp_summary])

                # ── 手术区：大模型(嫁接源)选择联动 ──
                def _on_exp_large_model_change(src_model, large_model, large_path_text):
                    """嫁接时选择大模型 → 显示源 vs 大模型架构对比"""
                    # 优先使用文本框路径
                    effective = (large_path_text or "").strip() or large_model
                    if not effective:
                        return "_选择大模型或输入路径后显示架构对比_"
                    try:
                        from core.model_editor import get_model_arch_brief
                        # 尝试解析路径
                        try:
                            large_info = get_model_arch_brief(_resolve_model_path(effective))
                        except Exception:
                            large_info = get_model_arch_brief(effective)
                        if src_model:
                            src_info = get_model_arch_brief(_resolve_model_path(src_model))
                            return f"**小模型（接收方）**: {src_info}\n\n**大模型（知识源）**: {large_info}"
                        return f"**大模型（知识源）**: {large_info}"
                    except Exception as e:
                        return f"⚠️ {e}"
                exp_large_model.change(
                    _on_exp_large_model_change, [exp_model, exp_large_model, exp_large_path], [exp_graft_info])
                exp_large_path.change(
                    _on_exp_large_model_change, [exp_model, exp_large_model, exp_large_path], [exp_graft_info])

                def _expansion_dispatch(model, output, method_label,
                                         depth_layers, depth_strat, noise,
                                         target_h, target_inter,
                                         hyb_layers, hyb_hidden, hyb_inter,
                                         large_model, large_path_text, graft_n,
                                         new_tokens, vocab_size,
                                         moe_experts, moe_topk, moe_layers_str_val):
                    if not model:
                        return "", "❌ 请选择源模型"
                    method = (method_label or "").split(" — ")[0].strip()
                    model_path = _resolve_model_path(model)
                    out = str(output or f"{method}_expanded")

                    try:
                        if method == "depth":
                            return expansion_depth_submit(
                                "本机模型", "", model_path, out,
                                str(depth_strat or "repeat_middle"),
                                int(depth_layers or 4), float(noise or 0.01),
                            )
                        elif method == "width":
                            return expansion_width_submit(
                                "本机模型", "", model_path, out,
                                int(target_h or 0), int(target_inter or 0),
                                float(noise or 0.01),
                            )
                        elif method == "hybrid":
                            return expansion_hybrid_submit(
                                "本机模型", "", model_path, out,
                                int(hyb_layers or 0), int(hyb_hidden or 0),
                                int(hyb_inter or 0), str(depth_strat or "repeat_middle"),
                                float(noise or 0.01),
                            )
                        elif method == "graft":
                            # 优先使用文本框路径（支持任意 HF 路径）
                            effective_large = (large_path_text or "").strip()
                            if not effective_large and large_model:
                                effective_large = _resolve_model_path(large_model)
                            if not effective_large:
                                return "", "❌ 请选择大模型或输入 HF 模型路径（知识源）"
                            # 判断是 HF ID 还是本地路径
                            from pathlib import Path as _P
                            if _P(effective_large).is_dir():
                                return expansion_graft_submit(
                                    "本机模型", "", model_path,
                                    "本机模型", "", effective_large,
                                    out, "", int(graft_n or 4), float(noise or 0.005),
                                )
                            else:
                                # 可能是 HF model ID
                                return expansion_graft_submit(
                                    "本机模型", "", model_path,
                                    "HF 模型ID", effective_large, "",
                                    out, "", int(graft_n or 4), float(noise or 0.005),
                                )
                        elif method == "vocab":
                            return expansion_vocab_submit(
                                "本机模型", "", model_path, out,
                                str(new_tokens or ""), int(vocab_size or 0), "mean",
                            )
                        elif method == "moe":
                            return expansion_moe_submit(
                                "本机模型", "", model_path, out,
                                int(moe_experts or 4), int(moe_topk or 2),
                                str(moe_layers_str_val or ""), float(noise or 0.01),
                            )
                        else:
                            return "", f"❌ 未知操作: {method}"
                    except Exception as e:
                        return "", f"❌ {e}"

                btn_expand.click(
                    _expansion_dispatch,
                    [exp_model, exp_output, exp_method,
                     exp_depth_layers, exp_depth_strat, exp_noise,
                     exp_target_h, exp_target_inter,
                     exp_hyb_layers, exp_hyb_hidden, exp_hyb_inter,
                     exp_large_model, exp_large_path, exp_graft_n,
                     exp_new_tokens, exp_vocab_size,
                     exp_moe_experts, exp_moe_topk, exp_moe_layers_str],
                    [exp_task_id, exp_msg],
                )

            # ════════════════════════════════════════════════════
            # 🧠 差分神经元系统 (统一: MoLoRA + 蜂群)
            # ════════════════════════════════════════════════════
            with gr.Accordion("🧠 差分神经元 — 多专家/多人格系统", open=False):
                gr.Markdown(
                    "### 一套系统，两种玩法\n\n"
                    "| | 🔬 MoLoRA 模式 | 🐝 蜂群模式 |\n"
                    "|---|---|---|\n"
                    "| **原理** | 一个基座内多组 LoRA 专家 + 内置门控 | 小门控 + 多个独立专家模型 |\n"
                    "| **输出** | 训练后合并 → **一个标准模型文件** | 推理时按需加载, **显存只占一个** |\n"
                    "| **兼容** | Ollama / vLLM / llama.cpp 直接用 | 通过本系统路由推理 |\n"
                    "| **适合** | 想训出一个全能模型 | 已有多个专业模型, 组团协作 |\n"
                    "| **专家** | 共享基座, 不同 LoRA 分支 | **可以是完全不同的模型** |\n"
                    "| **显存** | 一个模型的量 | 门控(~0.5GB) + 1个专家 |"
                )

                from core.neuron_system import (
                    NeuronConfig, NeuronDef, DOMAIN_CHOICES, NEURON_COLORS,
                    DOMAIN_KEYWORDS,
                    save_config, load_config, list_configs,
                    format_route_preview, estimate_vram,
                )

                # 预设门控模型选项
                _GATEWAY_PRESETS = [
                    "Qwen/Qwen2.5-0.5B-Instruct",
                    "Qwen/Qwen2.5-1.5B-Instruct",
                    "Qwen/Qwen3-0.6B",
                    "Qwen/Qwen3-1.7B",
                    "HuggingFaceTB/SmolLM2-360M-Instruct",
                    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                ]

                # 预设专家模型推荐 (按领域)
                _EXPERT_PRESETS = {
                    "医学": ["FreedomIntelligence/HuatuoGPT2-7B", "google/medgemma-4b-it"],
                    "代码": ["deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct"],
                    "法律": ["ShengbinYue/DISC-LawLLM", "Qwen/Qwen2.5-7B-Instruct"],
                    "金融": ["Duxiaoman-DI/XuanYuan-FinX1-Preview", "Qwen/Qwen2.5-7B-Instruct"],
                    "通用": ["Qwen/Qwen2.5-7B-Instruct", "01-ai/Yi-1.5-9B-Chat", "mistralai/Mistral-7B-Instruct-v0.3"],
                }

                _ns_state = gr.State(None)       # NeuronConfig
                _ns_swarm = gr.State(None)        # NeuronSwarm 实例

                # ── 模式选择 + 配置管理 ──
                with gr.Row():
                    ns_mode = gr.Radio(
                        label="模式",
                        choices=["🐝 蜂群模式 (多模型路由)", "🔬 MoLoRA 模式 (单模型训练)"],
                        value="🐝 蜂群模式 (多模型路由)", scale=3,
                        info="蜂群=已有多个模型想协同 | MoLoRA=从零训一个全能模型",
                    )
                with gr.Row():
                    ns_config_name = gr.Textbox(
                        label="配置名", value="my_neurons", scale=2,
                        info="保存到 data/neuron_configs/{名称}.json")
                    ns_config_list = gr.Dropdown(
                        label="已保存的配置", choices=list_configs(), scale=2,
                        info="选择后点加载")
                    btn_ns_refresh = gr.Button("🔄", scale=0, min_width=40)
                with gr.Row():
                    btn_ns_load = gr.Button("📂 加载配置", variant="secondary")
                    btn_ns_save = gr.Button("💾 保存配置", variant="primary")
                ns_status = gr.Markdown("等待配置...")

                # ── 神经元列表 ──
                ns_table = gr.Dataframe(
                    headers=["名称", "领域", "模型/数据", "关键词", "优先级"],
                    col_count=(5, "fixed"), interactive=False,
                    label="🧬 已注册神经元",
                )

                # ── 添加神经元 ──
                with gr.Accordion("➕ 添加神经元", open=True):
                    with gr.Row():
                        ns_name = gr.Textbox(
                            label="名称", placeholder="医学专家", scale=2,
                            info="神经元的显示名称")
                        ns_domain = gr.Dropdown(
                            label="领域", choices=DOMAIN_CHOICES,
                            value="通用", allow_custom_value=True, scale=1,
                            info="选择后自动填充关键词")
                        ns_priority = gr.Slider(
                            0, 10, value=5, step=1, label="优先级", scale=1,
                            info="越高越优先被路由到")

                    # 蜂群模式: 模型路径
                    with gr.Group(visible=True) as ns_swarm_fields:
                        with gr.Row():
                            ns_model_path = gr.Dropdown(
                                label="🐝 专家模型",
                                choices=_list_all_model_dirs() + _EXPERT_PRESETS.get("通用", []),
                                allow_custom_value=True, scale=3,
                                info="选择本地模型或输入 HuggingFace ID")
                            ns_4bit = gr.Checkbox(
                                label="4bit 量化", value=True, scale=1,
                                info="推荐开启, 省约 60% 显存")
                        ns_desc = gr.Textbox(
                            label="描述 (会作为系统提示词)",
                            placeholder="擅长诊断分析、用药建议和病理解读",
                            info="让专家知道自己是谁, 回答更专业")

                    # MoLoRA 模式: 数据集 + LoRA
                    with gr.Group(visible=False) as ns_molora_fields:
                        with gr.Row():
                            ns_datasets = gr.Dropdown(
                                label="🔬 训练数据集（可多选）", choices=_list_datasets(),
                                multiselect=True, scale=2,
                                info="该神经元专用的训练数据")
                            ns_lora = gr.Dropdown(
                                label="已有 LoRA（热启动, 可选）",
                                choices=_list_all_model_dirs(), scale=2,
                                info="用现有 LoRA 初始化, 加速收敛")

                    ns_keywords = gr.Textbox(
                        label="路由关键词 (逗号分隔)",
                        placeholder="诊断, 症状, 治疗, 药物, 临床",
                        info="选择领域后自动填充, 也可手动编辑")
                    with gr.Row():
                        btn_ns_add = gr.Button("➕ 添加神经元", variant="primary")
                        ns_del_target = gr.Dropdown(
                            label="选择要删除的神经元", choices=[], scale=2,
                            info="从已注册的神经元中选择")
                        btn_ns_remove = gr.Button("🗑️ 删除", variant="stop")

                # ── 蜂群: 门控设置 ──
                with gr.Accordion("🚪 门控路由器 (蜂群模式)", open=False, visible=True) as ns_gw_section:
                    with gr.Row():
                        ns_gateway = gr.Dropdown(
                            label="门控模型",
                            choices=_GATEWAY_PRESETS,
                            value="Qwen/Qwen2.5-0.5B-Instruct",
                            allow_custom_value=True,
                            scale=3,
                            info="常驻显存的小模型, 只做路由判断 (~300-700MB)")
                        ns_route_mode = gr.Dropdown(
                            label="路由策略",
                            choices=[
                                ("🔀 混合 (关键词粗筛 + LLM精排, 推荐)", "hybrid"),
                                ("🤖 纯LLM (全部由门控模型决策)", "gateway_llm"),
                                ("⚡ 纯关键词 (最快, 不需要门控模型)", "keyword"),
                            ],
                            value="hybrid", scale=2,
                            info="hybrid 兼顾速度和准确度")
                    with gr.Row():
                        ns_cache_cpu = gr.Checkbox(
                            label="CPU 缓存 (卸载而非释放)", value=True, scale=1,
                            info="切回同一专家时秒级恢复, 消耗 CPU 内存")
                        ns_max_cpu = gr.Slider(
                            0, 3, value=1, step=1,
                            label="CPU 最多缓存模型数", scale=1,
                            info="0=不缓存, 每个约占 4-8GB CPU 内存")
                    ns_vram_est = gr.Markdown("")

                # ── MoLoRA: 训练参数 ──
                with gr.Accordion("⚙️ MoLoRA 训练参数", open=False, visible=False) as ns_ml_section:
                    with gr.Row():
                        ns_base_model = gr.Dropdown(
                            label="基座模型", choices=_list_all_model_dirs(),
                            allow_custom_value=True, scale=2,
                            info="所有神经元共享同一个基座")
                        ns_rank = gr.Slider(
                            4, 64, value=16, step=4, label="专家 Rank", scale=1,
                            info="每个专家的 LoRA rank, 16=轻量 32=推荐 64=高质量")
                        ns_topk = gr.Slider(
                            1, 4, value=2, step=1, label="Top-K 路由", scale=1,
                            info="每次激活几个专家, 1=最稀疏 2=协作")
                    with gr.Row():
                        ns_lr = gr.Number(value=2e-4, label="学习率", info="通常 1e-4 ~ 5e-4")
                        ns_epochs = gr.Number(value=1.0, label="训练轮次", info="<1k条用3轮, >5k用1轮")
                        ns_maxseq = gr.Slider(
                            512, 8192, value=2048, step=256, label="最大序列长度")

                # ── 操作按钮 ──
                with gr.Row():
                    btn_ns_train = gr.Button("🚀 MoLoRA 训练", variant="primary", visible=False)
                    btn_ns_start = gr.Button("🚀 启动蜂群", variant="primary", visible=True)
                    btn_ns_stop = gr.Button("⏹️ 关闭蜂群", variant="stop", visible=True)
                ns_action_msg = gr.Markdown("")

                # ── 路由预览 ──
                with gr.Accordion("🔍 路由预览 (不需要加载模型)", open=False):
                    ns_test_q = gr.Textbox(
                        label="测试问题", lines=2,
                        placeholder="患者头痛伴视力模糊，应如何诊断？",
                        info="输入问题, 看看会路由到哪个专家")
                    btn_ns_test = gr.Button("🔍 预览路由")
                    ns_test_result = gr.Markdown("")

                # ── 蜂群推理 ──
                with gr.Accordion("💬 蜂群推理", open=False, visible=True) as ns_infer_section:
                    ns_query = gr.Textbox(
                        label="💬 提问 (自动路由到对应专家)",
                        placeholder="请帮我分析这个问题...", lines=3,
                        info="蜂群启动后, 问题会自动发给最合适的专家")
                    with gr.Row():
                        ns_max_tok = gr.Slider(
                            64, 2048, value=512, step=64, label="最大生成长度")
                        ns_temp = gr.Slider(
                            0, 1.5, value=0.7, step=0.1, label="温度",
                            info="0=确定性 0.7=推荐 1.0+=创意")
                    btn_ns_query = gr.Button("🧠 提问 (自动路由)", variant="primary")
                    ns_answer = gr.Markdown("")

                # ═══════════════════════════════════════
                # 事件绑定
                # ═══════════════════════════════════════

                # 模式切换 → 显示/隐藏
                def _ns_switch_mode(mode_str):
                    is_swarm = "蜂群" in mode_str
                    return (
                        gr.update(visible=is_swarm),       # ns_swarm_fields
                        gr.update(visible=not is_swarm),    # ns_molora_fields
                        gr.update(visible=is_swarm),        # ns_gw_section
                        gr.update(visible=not is_swarm),    # ns_ml_section
                        gr.update(visible=not is_swarm),    # btn_ns_train
                        gr.update(visible=is_swarm),        # btn_ns_start
                        gr.update(visible=is_swarm),        # btn_ns_stop
                        gr.update(visible=is_swarm),        # ns_infer_section
                    )
                ns_mode.change(
                    _ns_switch_mode, [ns_mode],
                    [ns_swarm_fields, ns_molora_fields, ns_gw_section, ns_ml_section,
                     btn_ns_train, btn_ns_start, btn_ns_stop, ns_infer_section])

                # 选领域 → 自动填关键词 + 推荐模型
                def _ns_domain_change(domain, current_kw, mode_str):
                    kw_list = DOMAIN_KEYWORDS.get(domain, [])
                    new_kw = ", ".join(kw_list[:10]) if kw_list and not current_kw else current_kw
                    # 蜂群模式: 更新推荐模型列表
                    is_swarm = "蜂群" in (mode_str or "")
                    if is_swarm:
                        presets = _EXPERT_PRESETS.get(domain, _EXPERT_PRESETS.get("通用", []))
                        model_choices = _list_all_model_dirs() + presets
                        return new_kw, gr.update(choices=model_choices)
                    return new_kw, gr.update()
                ns_domain.change(
                    _ns_domain_change, [ns_domain, ns_keywords, ns_mode],
                    [ns_keywords, ns_model_path])

                # 刷新 (所有下拉框)
                def _ns_do_refresh():
                    models = _list_all_model_dirs()
                    return (gr.update(choices=list_configs()),
                            gr.update(choices=models),
                            gr.update(choices=_list_datasets()),
                            gr.update(choices=models),
                            gr.update(choices=models + _EXPERT_PRESETS.get("通用", [])))
                btn_ns_refresh.click(
                    _ns_do_refresh, None,
                    [ns_config_list, ns_base_model, ns_datasets, ns_lora, ns_model_path])

                # 更新删除下拉
                def _ns_update_del_dropdown(config):
                    if not config or not config.neurons:
                        return gr.update(choices=[], value=None)
                    names = [n.name for n in config.neurons]
                    return gr.update(choices=names, value=names[-1])

                # 表格数据
                def _ns_tbl(config, mode_str):
                    if not config: return []
                    is_swarm = "蜂群" in (mode_str or "")
                    rows = []
                    for n in config.neurons:
                        if is_swarm:
                            info = n.model_path or "-"
                        else:
                            ds = ", ".join(Path(d).name for d in n.datasets[:2])
                            info = ds or (Path(n.lora_path).name if n.lora_path else "-")
                        rows.append([n.name, n.domain, info,
                                     ", ".join(n.keywords[:4]) or "-", n.priority])
                    return rows

                # 加载配置 → 恢复全部 UI 状态
                def _ns_load(name_sel, mode_str):
                    if not name_sel:
                        return (None, "⚠️ 选择配置后点加载", [],
                                gr.update(), gr.update(), gr.update(),
                                gr.update(), gr.update(), gr.update())
                    cfg = load_config(name_sel)
                    if not cfg:
                        return (None, f"❌ 配置不存在: {name_sel}", [],
                                gr.update(), gr.update(), gr.update(),
                                gr.update(), gr.update(), gr.update())

                    tbl = _ns_tbl(cfg, mode_str)
                    del_choices = [n.name for n in cfg.neurons]

                    # 恢复模式
                    if cfg.mode == "molora":
                        mode_val = "🔬 MoLoRA 模式 (单模型训练)"
                    else:
                        mode_val = "🐝 蜂群模式 (多模型路由)"

                    return (
                        cfg,
                        f"✅ 已加载: {cfg.name} ({len(cfg.neurons)} 神经元, 模式={cfg.mode})",
                        tbl,
                        gr.update(value=cfg.name),                    # ns_config_name
                        gr.update(value=mode_val),                     # ns_mode
                        gr.update(value=cfg.gateway_model or "Qwen/Qwen2.5-0.5B-Instruct"),  # ns_gateway
                        gr.update(value=cfg.route_mode or "hybrid"),   # ns_route_mode
                        gr.update(choices=del_choices, value=del_choices[-1] if del_choices else None),  # ns_del_target
                        estimate_vram(cfg) if cfg.mode == "swarm" and cfg.neurons else "",  # ns_vram_est
                    )
                btn_ns_load.click(
                    _ns_load, [ns_config_list, ns_mode],
                    [_ns_state, ns_status, ns_table,
                     ns_config_name, ns_mode, ns_gateway, ns_route_mode,
                     ns_del_target, ns_vram_est])

                # 保存配置
                def _ns_save(config, name, mode_str, gateway, route_mode,
                             cache_cpu, max_cpu, base_model, rank, topk, lr, epochs, maxseq):
                    if config is None:
                        config = NeuronConfig()
                    config.name = name or "my_neurons"
                    config.mode = "swarm" if "蜂群" in (mode_str or "") else "molora"
                    config.gateway_model = gateway or ""
                    config.route_mode = route_mode or "hybrid"
                    config.cache_in_cpu = bool(cache_cpu)
                    config.max_cpu_cache = int(max_cpu or 1)
                    config.base_model = _resolve_model_path(base_model) if base_model else ""
                    config.rank = int(rank or 16)
                    config.top_k = int(topk or 2)
                    config.lr = float(lr or 2e-4)
                    config.epochs = float(epochs or 1.0)
                    config.max_seq_len = int(maxseq or 2048)
                    save_config(config)
                    # 刷新配置列表
                    return (config,
                            f"✅ 已保存: {config.name} (模式={config.mode})",
                            gr.update(choices=list_configs(), value=config.name))
                btn_ns_save.click(
                    _ns_save,
                    [_ns_state, ns_config_name, ns_mode, ns_gateway, ns_route_mode,
                     ns_cache_cpu, ns_max_cpu, ns_base_model, ns_rank, ns_topk,
                     ns_lr, ns_epochs, ns_maxseq],
                    [_ns_state, ns_status, ns_config_list])

                # 添加神经元
                def _ns_add(config, mode_str, name, domain, priority,
                            model_path, q4bit, desc, datasets, lora, keywords):
                    if config is None:
                        config = NeuronConfig()
                    if not name:
                        return (config, "❌ 名称不能为空",
                                _ns_tbl(config, mode_str), "", gr.update())
                    if any(n.name == name for n in config.neurons):
                        return (config, f"❌ 已存在同名神经元: {name}",
                                _ns_tbl(config, mode_str), "", gr.update())

                    is_swarm = "蜂群" in (mode_str or "")
                    kw = [k.strip() for k in (keywords or "").split(",") if k.strip()]
                    color = NEURON_COLORS[len(config.neurons) % len(NEURON_COLORS)]

                    neuron = NeuronDef(
                        name=name, domain=domain or "通用", priority=int(priority or 5),
                        keywords=kw, color=color, enabled=True,
                    )
                    if is_swarm:
                        if not model_path:
                            return (config, "❌ 蜂群模式需要指定专家模型",
                                    _ns_tbl(config, mode_str), "", gr.update())
                        neuron.model_path = model_path
                        neuron.quantize_4bit = bool(q4bit)
                        neuron.description = desc or ""
                    else:
                        from core import DATASETS_DIR
                        neuron.datasets = [str(DATASETS_DIR / f) for f in (datasets or []) if f]
                        neuron.lora_path = _resolve_model_path(lora) if lora else ""

                    config.neurons.append(neuron)
                    est = estimate_vram(config) if is_swarm else ""
                    del_names = [n.name for n in config.neurons]
                    return (config,
                            f"✅ 已添加: {name} [{domain}]",
                            _ns_tbl(config, mode_str), est,
                            gr.update(choices=del_names, value=name))

                btn_ns_add.click(
                    _ns_add,
                    [_ns_state, ns_mode, ns_name, ns_domain, ns_priority,
                     ns_model_path, ns_4bit, ns_desc, ns_datasets, ns_lora, ns_keywords],
                    [_ns_state, ns_status, ns_table, ns_vram_est, ns_del_target])

                # 删除指定神经元
                def _ns_remove(config, mode_str, target_name):
                    if not config or not config.neurons:
                        return (config, "⚠️ 没有神经元可删除",
                                _ns_tbl(config, mode_str), "", gr.update(choices=[], value=None))
                    if target_name:
                        config.neurons = [n for n in config.neurons if n.name != target_name]
                        msg = f"🗑️ 已删除: {target_name}"
                    else:
                        removed = config.neurons.pop()
                        msg = f"🗑️ 已删除: {removed.name}"

                    est = estimate_vram(config) if config.neurons and "蜂群" in (mode_str or "") else ""
                    del_names = [n.name for n in config.neurons]
                    return (config, msg, _ns_tbl(config, mode_str), est,
                            gr.update(choices=del_names,
                                      value=del_names[-1] if del_names else None))
                btn_ns_remove.click(
                    _ns_remove, [_ns_state, ns_mode, ns_del_target],
                    [_ns_state, ns_status, ns_table, ns_vram_est, ns_del_target])

                # 路由预览
                def _ns_test(config, query, mode_str):
                    if not config or not config.neurons or not query:
                        return "⚠️ 请添加神经元并输入测试问题"
                    m = "swarm" if "蜂群" in (mode_str or "") else "molora"
                    return format_route_preview(query, config.neurons, m)
                btn_ns_test.click(_ns_test, [_ns_state, ns_test_q, ns_mode], ns_test_result)

                # MoLoRA 训练
                def _ns_train(config, mode_str, base_model, rank, topk, lr, epochs, maxseq):
                    if "MoLoRA" not in (mode_str or ""):
                        return "", "❌ 请切换到 MoLoRA 模式"
                    if not config or len(config.neurons) < 2:
                        return "", "❌ 至少需要 2 个神经元"
                    config.mode = "molora"
                    config.base_model = _resolve_model_path(base_model) if base_model else ""
                    if not config.base_model:
                        return "", "❌ 请选择基座模型"
                    config.rank = int(rank or 16)
                    config.top_k = int(topk or 2)
                    config.lr = float(lr or 2e-4)
                    config.epochs = float(epochs or 1.0)
                    config.max_seq_len = int(maxseq or 2048)

                    from core.neuron_system import train_molora
                    names = ", ".join(n.name for n in config.neurons)
                    def _run(task):
                        return train_molora(config, task=task)
                    tid = task_queue.submit(f"🧠 MoLoRA: {names}", _run)
                    return tid, f"✅ MoLoRA 训练已提交: {tid}\n  神经元: {names}"

                btn_ns_train.click(
                    _ns_train,
                    [_ns_state, ns_mode, ns_base_model, ns_rank, ns_topk,
                     ns_lr, ns_epochs, ns_maxseq],
                    [gr.Textbox(visible=False), ns_action_msg])

                # 蜂群启动
                def _ns_start(config, swarm_inst, gateway, route_mode, cache_cpu, max_cpu):
                    if not config or not config.neurons:
                        return swarm_inst, "❌ 请先添加专家"
                    config.mode = "swarm"
                    config.gateway_model = gateway or ""
                    config.route_mode = route_mode or "hybrid"
                    config.cache_in_cpu = bool(cache_cpu)
                    config.max_cpu_cache = int(max_cpu or 1)
                    from core.neuron_system import NeuronSwarm
                    swarm = NeuronSwarm()
                    swarm.init(config)
                    try:
                        swarm.start()
                        return swarm, swarm.get_status_md()
                    except Exception as e:
                        return swarm_inst, f"❌ 启动失败: {e}"

                btn_ns_start.click(
                    _ns_start,
                    [_ns_state, _ns_swarm, ns_gateway, ns_route_mode, ns_cache_cpu, ns_max_cpu],
                    [_ns_swarm, ns_action_msg])

                def _ns_stop(swarm_inst):
                    if swarm_inst: swarm_inst.shutdown()
                    return None, "⏹️ 蜂群已关闭"
                btn_ns_stop.click(_ns_stop, [_ns_swarm], [_ns_swarm, ns_action_msg])

                # 蜂群推理
                def _ns_do_query(swarm_inst, query, max_tok, temp):
                    if not swarm_inst or not swarm_inst._started:
                        return "❌ 请先启动蜂群"
                    if not query: return "⚠️ 请输入问题"
                    try:
                        r = swarm_inst.query(query, max_new_tokens=int(max_tok or 512),
                                             temperature=float(temp or 0.7))
                        if "error" in r: return f"❌ {r['error']}"
                        t = r["timing"]
                        sc = " → ".join(f"{n}({s:.1f})" for n, s in r["route_scores"][:3])
                        return (f"### 🧠 {r['expert']} [{r['domain']}]\n\n{r['answer']}\n\n---\n"
                                f"⏱️ 路由 {t['route_ms']}ms → 加载 {t['load_ms']}ms → "
                                f"生成 {t['gen_ms']}ms = {t['total_ms']}ms | 📊 {sc}")
                    except Exception as e:
                        return f"❌ {e}"
                btn_ns_query.click(
                    _ns_do_query, [_ns_swarm, ns_query, ns_max_tok, ns_temp], ns_answer)

            # ════════════════════════════════════════════════════
            # ★ 核心联动: 选择模型自动加载（移到此处，exp_model 已定义）
            # ════════════════════════════════════════════════════
            _me_load_outputs = [
                me_model_card,
                me_quick_actions, me_summary,
                me_ready_badge, _active_edit_model,
                me_name, me_author, me_desc, me_tags, me_usecase,
                me_system_prompt, me_sp_preview,
                me_temp, me_top_p, me_top_k,
                me_repeat_penalty, me_max_tokens, me_stop_seq,
                me_kb_list, me_kb_status,
                me_safety_on, me_refusal, me_content_filter,
                me_disclaimer_on, me_disclaimer,
                me_bake_checklist, me_export_name,
                exp_model, exp_summary,
            ]
            me_model.change(_me_load_all, [me_model], _me_load_outputs)

            # 切换 Tab 时刷新模型列表
            def _on_forge_tab_select():
                _invalidate_loras_cache()
                models = _list_all_model_dirs()
                return (gr.update(choices=models), gr.update(choices=models),
                        gr.update(choices=models), gr.update(choices=models),
                        gr.update(choices=models))
            tab_forge.select(_on_forge_tab_select, None,
                             [me_model, exp_model, exp_large_model, ns_base_model, ns_lora])

        # ================================================================
        # ================================================================
        #  Tab 4: 🧪 评测
        # ================================================================
        with gr.Tab("🧪 评测") as tab_eval:
            gr.Markdown("### 训练效果验证：快速评测 + 实时对话测试")

            # ---- 快速评测 ----
            with gr.Accordion("📊 快速评测", open=True):
                gr.Markdown("用内置题集自动评分，快速判断模型质量")
                with gr.Row():
                    bench_model_dir = gr.Textbox(label="模型目录（HF ID 或本地合并目录）", value="", scale=2)
                    bench_model_picker = gr.Dropdown(
                        label="快速选择已有模型",
                        choices=_list_all_model_dirs(), value=None, scale=1,
                    )
                with gr.Row():
                    bench_preset = gr.Dropdown(label="题集", choices=[
                        ("基础10题", "basic_10"), ("推理10题", "reason_10"), ("格式10题", "format_10")
                    ], value="basic_10", scale=1)
                    btn_bench_from_editor = gr.Button("🔗 从编辑器导入", variant="secondary", scale=1)
                    btn_bench = gr.Button("▶️ 运行评测", variant="primary", scale=1)
                def _pick_bench_model(name):
                    if not name: return gr.update()
                    clean = _extract_model_name_from_label(name)
                    return gr.update(value=str(LORAS_DIR / clean))
                bench_model_picker.change(_pick_bench_model, bench_model_picker, bench_model_dir)
                bench_task_id = gr.Textbox(visible=False)
                bench_msg = gr.Textbox(label="状态", interactive=False)
                def _bench_from_editor(editor_model):
                    if not editor_model:
                        return gr.update(), "⚠️ 编辑器未选择模型"
                    clean = _extract_model_name_from_label(editor_model)
                    return gr.update(value=str(LORAS_DIR / clean)), f"✅ 已导入: {clean}"
                btn_bench_from_editor.click(
                    _bench_from_editor, [me_model], [bench_model_dir, bench_msg])
                btn_bench.click(bench_submit, [bench_model_dir, bench_preset], [bench_task_id, bench_msg])

            # ---- 聊天测试 ----
            with gr.Accordion("💬 聊天测试", open=True):
                gr.Markdown(
                    "载入模型进行实时对话，直观感受训练效果。\n"
                    "支持 HF 模型、合并模型、LoRA 适配器、GGUF 文件。\n\n"
                    "> 💡 在 **模型编辑** Tab 设置了人设/知识库？点「🔗 从编辑器导入」自动填充。"
                )
                with gr.Row():
                    chat_model_path = gr.Textbox(label="模型路径（HF ID / 本地目录 / GGUF文件）", scale=2)
                    chat_model_picker = gr.Dropdown(
                        label="快速选择 data/loras",
                        choices=_list_all_model_dirs(), value=None, scale=1,
                    )
                    chat_lora = gr.Textbox(label="LoRA路径（可选）", value="", placeholder="留空=自动检测", scale=1)
                with gr.Row():
                    chat_backend = gr.Dropdown(label="后端", choices=["auto", "hf", "gguf"], value="auto", scale=1)
                    btn_chat_from_editor = gr.Button("🔗 从编辑器导入", variant="secondary", scale=1,
                                                      elem_id="btn-chat-from-editor")
                    btn_chat_load = gr.Button("📦 载入模型", variant="primary", scale=1)
                    btn_chat_unload = gr.Button("🗑️ 卸载", variant="stop", scale=1)
                    chat_status = gr.Textbox(label="状态", interactive=False, scale=2)

                # 快速选择 → 填入路径 + 自动读取 profile 系统提示词
                def _pick_chat_model(name):
                    if not name:
                        return gr.update(), gr.update(), gr.update()
                    clean = _extract_model_name_from_label(name)
                    mp = str(LORAS_DIR / clean)
                    sys_prompt = "You are a helpful assistant."
                    temp_val = 0.7
                    try:
                        from core.model_editor import load_profile
                        profile = load_profile(mp)
                        sp = profile.get("system_prompt", "")
                        if sp:
                            sys_prompt = sp
                        params = profile.get("parameters", {})
                        if params.get("temperature") is not None:
                            temp_val = params["temperature"]
                    except Exception:
                        pass
                    return gr.update(value=mp), gr.update(value=sys_prompt), gr.update(value=temp_val)

                # 从编辑器导入 → 填入路径 + 系统提示词 + 温度 + LoRA
                def _chat_import_from_editor(editor_model):
                    if not editor_model:
                        return (gr.update(), gr.update(), gr.update(),
                                gr.update(), "⚠️ 编辑器未选择模型，请先在「模型编辑」Tab 选择")
                    try:
                        clean = _extract_model_name_from_label(editor_model)
                        mp = str(LORAS_DIR / clean)
                        from core.model_editor import load_profile, generate_system_prompt_with_knowledge
                        profile = load_profile(mp)
                        # 优先使用完整提示词（含知识库+安全）
                        try:
                            full_sp = generate_system_prompt_with_knowledge(mp)
                        except Exception:
                            full_sp = ""
                        sys_prompt = full_sp or profile.get("system_prompt", "") or "You are a helpful assistant."
                        params = profile.get("parameters", {})
                        temp_val = params.get("temperature", 0.7)
                        # 检测 LoRA
                        from pathlib import Path
                        lora_path = ""
                        if (Path(mp) / "adapter_config.json").exists():
                            lora_path = mp
                        name = profile.get("name", clean)
                        kb_count = len(profile.get("knowledge_docs", []))
                        safety_on = profile.get("safety", {}).get("enabled", False)
                        parts = [f"✅ 已导入「{name}」"]
                        if full_sp and full_sp != profile.get("system_prompt", ""):
                            parts.append("含知识库/安全规则")
                        if kb_count:
                            parts.append(f"{kb_count}个知识文档")
                        if safety_on:
                            parts.append("安全护栏已启用")
                        msg = " | ".join(parts)
                        return (gr.update(value=mp), gr.update(value=sys_prompt),
                                gr.update(value=temp_val), gr.update(value=lora_path), msg)
                    except Exception as e:
                        return (gr.update(), gr.update(), gr.update(),
                                gr.update(), f"❌ 导入失败: {e}")

                with gr.Row():
                    chat_sys = gr.Textbox(label="系统提示词", value="You are a helpful assistant.", scale=3,
                                          info="选择模型后自动从 profile 读取，也可手动修改")
                    chat_temp = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="温度")
                    chat_max = gr.Number(label="最大生成长度", value=512, precision=0)

                # 绑定事件（chat_sys/chat_temp 已定义）
                chat_model_picker.change(
                    _pick_chat_model, [chat_model_picker],
                    [chat_model_path, chat_sys, chat_temp])
                btn_chat_from_editor.click(
                    _chat_import_from_editor, [me_model],
                    [chat_model_path, chat_sys, chat_temp, chat_lora, chat_status])

                chatbot = _chatbot_messages(label="对话", height=400)
                with gr.Row():
                    chat_input = gr.Textbox(label="输入", placeholder="输入你的消息...", scale=4, lines=1)
                    btn_chat_send = gr.Button("发送", variant="primary", scale=1)
                    btn_chat_clear = gr.Button("清空", scale=1)

                from core.chat_engine import chat_engine
                def _chat_load(mp, lora, backend):
                    if not mp or not mp.strip():
                        return "❌ 请输入模型路径或从下拉列表选择"
                    return chat_engine.load_model(mp.strip(), lora_path=lora.strip() or None, backend=backend)

                def _chat_unload():
                    return chat_engine.unload()

                def _chat_send(message, history, sys_prompt, temp, max_tok):
                    if not message or not message.strip():
                        return history or [], ""
                    history = list(history or [])
                    try:
                        reply = chat_engine.chat(
                            message=message.strip(),
                            history=[{"role": m["role"], "content": m["content"]} for m in history],
                            system_prompt=sys_prompt or "",
                            temperature=float(temp),
                            max_new_tokens=int(max_tok or 512),
                        )
                    except Exception as e:
                        reply = f"❌ 推理异常: {e}"
                    history.append({"role": "user", "content": message.strip()})
                    history.append({"role": "assistant", "content": reply})
                    return history, ""

                btn_chat_load.click(_chat_load, [chat_model_path, chat_lora, chat_backend], chat_status)
                btn_chat_unload.click(_chat_unload, None, chat_status)
                btn_chat_send.click(_chat_send, [chat_input, chatbot, chat_sys, chat_temp, chat_max], [chatbot, chat_input])
                chat_input.submit(_chat_send, [chat_input, chatbot, chat_sys, chat_temp, chat_max], [chatbot, chat_input])
                btn_chat_clear.click(lambda: ([], ""), None, [chatbot, chat_input])

                # RLHF 反馈按钮 — 修复: 用 chat_model_path 的值而非组件对象
                with gr.Row():
                    btn_thumbsup = gr.Button("👍 好回复", size="sm")
                    btn_thumbsdown = gr.Button("👎 差回复", size="sm")
                    rlhf_msg = gr.Textbox(label="反馈", interactive=False, scale=2)

                def _record_feedback_pos(history, model_path_val):
                    return _do_record_feedback("positive", history, model_path_val)
                def _record_feedback_neg(history, model_path_val):
                    return _do_record_feedback("negative", history, model_path_val)
                def _do_record_feedback(action, history, model_path_val):
                    if not history or len(history) < 2:
                        return "❌ 需要至少一轮对话才能反馈"
                    last_user = next(
                        (m["content"] for m in reversed(history) if m.get("role") == "user"), ""
                    )
                    last_asst = next(
                        (m["content"] for m in reversed(history) if m.get("role") == "assistant"), ""
                    )
                    if not last_user or not last_asst:
                        return "❌ 未找到有效的对话记录"
                    try:
                        _feedback.record(action, last_user, last_asst,
                                         model_name=str(model_path_val or "unknown"))
                        return f"✅ 已记录 {action} 反馈"
                    except Exception as e:
                        return f"❌ 记录失败: {e}"

                btn_thumbsup.click(_record_feedback_pos, [chatbot, chat_model_path], rlhf_msg)
                btn_thumbsdown.click(_record_feedback_neg, [chatbot, chat_model_path], rlhf_msg)

            # 聊天提取连接（在锻造引擎 tab 定义的）
            def _extract_from_chat(history, name):
                if not history or len(history) < 2: return "❌ 需要至少一轮对话"
                path = _forge.extract_training_from_chat(history, output_name=name or "chat_extracted")
                if path:
                    _invalidate_ds_cache()
                    return f"✅ 提取成功: {path.name}"
                return "❌ 提取失败"

            # 切换到评测 Tab 时刷新模型列表
            def _on_eval_tab_select():
                _invalidate_loras_cache()
                items = _list_all_model_dirs()
                return gr.update(choices=items), gr.update(choices=items)
            tab_eval.select(_on_eval_tab_select, None, [bench_model_picker, chat_model_picker])

        # ================================================================
        #  Tab 5: 🚀 部署
        # ================================================================
        with gr.Tab("🚀 部署") as tab_deploy:
            # ---- 合并 ----
            gr.Markdown("### 🔀 LoRA 合并")
            gr.Markdown("将训练好的 LoRA 适配器合并到基座模型，生成独立的完整模型")
            with gr.Row():
                base_for_merge = gr.Textbox(label="基座模型", value=config.get("default_base_model",""), scale=2)
                hf_cached_merge = gr.Dropdown(label="本机 HF 模型", choices=[], value=None, scale=1)
                btn_refresh_loras = gr.Button("🔄 刷新 LoRA", scale=1)
                btn_scan_models_merge = gr.Button("🔄 扫描模型", scale=1)
            loras = gr.CheckboxGroup(label="LoRA 适配器（来自 data/loras）", choices=_list_loras())
            def _refresh_loras_list():
                _invalidate_loras_cache()
                return gr.update(choices=_list_loras())
            btn_refresh_loras.click(_refresh_loras_list, None, loras)
            btn_scan_models_merge.click(fn=_refresh_local_hf_model_ids, inputs=None, outputs=hf_cached_merge, show_progress="full")
            hf_cached_merge.change(lambda x: gr.update(value=x), hf_cached_merge, base_for_merge)

            with gr.Row():
                merged_name = gr.Textbox(label="合并输出名", value="merged_model", scale=2)
                btn_merge = gr.Button("🔀 合并", variant="primary", scale=1)
            merge_task_id = gr.Textbox(visible=False)
            merge_msg = gr.Textbox(label="状态", interactive=False)
            btn_merge.click(merge_submit, [base_for_merge, loras, merged_name], [merge_task_id, merge_msg])

            # ---- 模型杂交融合 ----
            with gr.Accordion("🧬 模型杂交融合 — 原生张量级融合", open=False):
                gr.Markdown(
                    "将多个**完整模型**用高级算法融合，产生兼具多种能力的混血模型。\n"
                    "**纯 PyTorch 实现**，无需安装 mergekit。"
                )
                hybrid_method_tips = gr.Markdown(
                    "**SLERP** — 球面线性插值，两个模型间最平滑的过渡。需要 1 个模型 + 基座。\n"
                    "参数 `t`: 0=全基座, 0.5=均衡混合, 1=全目标模型"
                )
                with gr.Row():
                    hybrid_base = gr.Textbox(label="基座模型（HF ID 或本地路径）",
                        value=config.get("default_base_model",""), scale=2)
                    hybrid_method = gr.Dropdown(
                        label="融合算法",
                        choices=["slerp", "linear", "ties", "dare_ties",
                                 "dare_linear", "task_arithmetic", "frankenmerge"],
                        value="slerp", scale=1,
                    )

                # 动态方法说明
                def _update_method_tips(method):
                    tips = {
                        "slerp": "**SLERP** — 球面线性插值，两个模型间最平滑的过渡。需要 1 个模型 + 基座。\n参数 `t`: 0=全基座, 0.5=均衡混合, 1=全目标模型",
                        "linear": "**Linear** — 加权平均。简单稳定，适合多个同架构模型混合。\n每个模型可设不同权重（归一化后生效）",
                        "ties": "**TIES** — 修剪冗余参数变化 + 符号投票 + 合并。去除冲突，保留重要变化。\n`density`: 保留多少比例的参数变化。0.5=保留 50%",
                        "dare_ties": "**DARE-TIES** — 先随机丢弃一部分变化，再 TIES 融合。更激进的压缩。\n适合 3+ 个差异大的模型融合",
                        "dare_linear": "**DARE-Linear** — 随机丢弃 + 线性平均。比 DARE-TIES 温和",
                        "task_arithmetic": "**Task Arithmetic** — 计算每个模型的「任务向量」(= 模型 - 基座)，加权叠加。\n可以做减法（负权重 = 遗忘某种能力）",
                        "frankenmerge": "**Frankenmerge** — 从不同模型取不同层范围拼接。\n如: 模型A 的前 12 层 + 模型B 的后 12 层 = 混血怪物",
                    }
                    return tips.get(method, "")
                hybrid_method.change(_update_method_tips, hybrid_method, hybrid_method_tips)

                hybrid_loras = gr.CheckboxGroup(
                    label="参与融合的模型（完整模型，非 LoRA）", choices=_list_loras())

                # 通用参数
                with gr.Row():
                    hybrid_weight = gr.Slider(0.0, 2.0, value=0.5, step=0.05,
                        label="权重 t / 模型权重",
                        info="SLERP: 插值系数; 其他: 每个模型的权重 (>1 可超调)")
                    hybrid_density = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                        label="密度（TIES/DARE）",
                        info="保留多少比例的参数变化。越低=越激进")
                    hybrid_out = gr.Textbox(label="输出名", value="hybrid_model", scale=1)

                btn_hybrid = gr.Button("🧬 开始杂交融合", variant="primary", size="lg")
                hybrid_franken = gr.Textbox(visible=False, value="")  # kept for submit signature
                hybrid_task_id = gr.Textbox(visible=False)
                hybrid_msg = gr.Markdown("")

                btn_hybrid.click(
                    native_hybrid_submit,
                    [hybrid_base, hybrid_method, hybrid_loras,
                     hybrid_weight, hybrid_density, hybrid_out, hybrid_franken],
                    [hybrid_task_id, hybrid_msg],
                )

            def _on_deploy_tab_select():
                _invalidate_loras_cache()
                lora_items = _list_loras()
                all_items = _list_all_model_dirs()
                return gr.update(choices=lora_items), gr.update(choices=lora_items), gr.update(choices=all_items)
            # tab_deploy.select 移到所有组件定义之后（见 Ollama 区块后面）

            # ---- GGUF 导出 ----
            gr.Markdown("---\n### 📦 GGUF 导出")
            gr.Markdown("将**完整模型**（合并后的模型）量化导出为 GGUF 格式。如果选择 LoRA 适配器，会自动先合并再导出。")
            with gr.Row():
                gguf_model_dir = gr.Textbox(label="模型目录（完整模型路径或 LoRA 适配器路径）", value="", scale=3)
                gguf_merged_picker = gr.Dropdown(label="快速选择", choices=_list_all_model_dirs(), value=None, scale=2)
            with gr.Row():
                gguf_out = gr.Textbox(label="GGUF 输出名", value="my_model_gguf", scale=2)
                gguf_quant = gr.Dropdown(label="量化类型", choices=["F16","Q8_0","Q4_K_M","Q5_K_M","Q6_K"], value="Q4_K_M", scale=1)
                btn_gguf = gr.Button("📥 导出 GGUF", variant="primary", scale=1)
            def _pick_gguf_model(name):
                if not name: return gr.update()
                clean = _extract_model_name_from_label(name)
                return gr.update(value=str(LORAS_DIR / clean))
            gguf_merged_picker.change(_pick_gguf_model, gguf_merged_picker, gguf_model_dir)
            gguf_task_id = gr.Textbox(visible=False)
            gguf_msg = gr.Textbox(label="状态", interactive=False)
            btn_gguf.click(export_gguf_submit, [gguf_model_dir, gguf_out, gguf_quant], [gguf_task_id, gguf_msg])

            # ---- Ollama ----
            gr.Markdown("---\n### 🦙 Ollama 导入")
            gr.Markdown("将 GGUF 模型导入 Ollama，一键本地部署")
            with gr.Row():
                gguf_path = gr.Textbox(label="GGUF 路径", value="", scale=2)
                gguf_upload_path = gr.File(label="上传 GGUF", file_types=[".gguf"], scale=1)
                ollama_name = gr.Textbox(label="Ollama 模型名", value="my_model", scale=1)
            def _extract_gguf_path(f):
                if f is None: return gr.update(value="")
                if hasattr(f, "name"): return gr.update(value=str(f.name))
                if isinstance(f, dict): return gr.update(value=f.get("name", ""))
                return gr.update(value=str(f))
            gguf_upload_path.change(_extract_gguf_path, gguf_upload_path, gguf_path)
            btn_ollama = gr.Button("🚀 生成 Modelfile / 导入 Ollama", variant="primary")
            ollama_task_id = gr.Textbox(visible=False)
            ollama_msg = gr.Textbox(label="状态", interactive=False)
            btn_ollama.click(export_ollama_submit, [gguf_path, ollama_name], [ollama_task_id, ollama_msg])

            # 切换到部署 Tab 时刷新所有下拉列表（必须在所有组件定义之后）
            tab_deploy.select(_on_deploy_tab_select, None, [loras, hybrid_loras, gguf_merged_picker])

        # ================================================================
        #  Tab 6: 📋 任务中心
        # ================================================================
        with gr.Tab("📋 任务中心") as tab_monitor:
            gr.Markdown("### 实时监控所有任务 — 训练曲线 · 考试成绩 · 做题批改实况")
            with gr.Row():
                watch_id = gr.Dropdown(
                    label="选择任务", choices=[], allow_custom_value=True,
                    scale=3, info="选择要监控的任务（新任务自动出现）",
                )
                btn_refresh = gr.Button("🔄 刷新", scale=1)
                btn_cancel = gr.Button("⏹️ 取消任务", variant="stop", scale=1)
            with gr.Row():
                auto_refresh = gr.Checkbox(label="自动刷新 (3秒)", value=True, scale=1)
            with gr.Row():
                status = gr.Textbox(label="状态", interactive=False, scale=2)
                progress_bar = gr.Slider(0, 100, value=0, label="进度", interactive=False, scale=1)
            logs = gr.Textbox(label="日志（末尾200行）", lines=12, interactive=False)
            loss_plot = gr.HTML(label="📊 多功能图表（训练曲线 + 考试成绩）")

            # ── 断点续传 & 追加数据集 ──
            with gr.Accordion("🔄 断点续传 / 追加数据集", open=False):
                gr.Markdown(
                    "训练中断（OOM、手动取消、意外关闭）后，可从最近的 checkpoint 恢复。\n"
                    "也可以追加新数据集后继续训练。"
                )
                with gr.Row():
                    resume_lora = gr.Dropdown(
                        label="选择 LoRA（有断点的）", choices=_list_resumable_loras(),
                        allow_custom_value=True, scale=2,
                        info="只显示有 checkpoint 的 LoRA",
                    )
                    resume_ckpt = gr.Dropdown(
                        label="选择断点", choices=["最新"],
                        value="最新", scale=1,
                        info="默认从最新断点恢复",
                    )
                with gr.Row():
                    resume_extra_ds = gr.Dropdown(
                        label="追加数据集（可选）", choices=_list_datasets(),
                        multiselect=True, scale=2,
                        info="留空=只用原数据集恢复训练",
                    )
                    resume_epochs = gr.Number(
                        label="追加轮次", value=0, minimum=0, scale=1,
                        info="0=延续原设置",
                    )
                with gr.Row():
                    btn_resume = gr.Button("🔄 从断点恢复训练", variant="primary", scale=1)
                    btn_refresh_resume = gr.Button("🔃 刷新列表", scale=1)
                resume_msg = gr.Textbox(label="恢复状态", interactive=False)

                def _on_lora_select(lora_choice):
                    """选择 LoRA 后刷新断点列表"""
                    if not lora_choice:
                        return gr.update(choices=["最新"], value="最新")
                    clean = lora_choice.split(" (")[0].strip()
                    ckpts = _find_checkpoints(clean)
                    choices = ["最新"] + ckpts
                    return gr.update(choices=choices, value="最新")

                resume_lora.change(_on_lora_select, resume_lora, resume_ckpt)

                def _refresh_resume_lists():
                    return (
                        gr.update(choices=_list_resumable_loras()),
                        gr.update(choices=_list_datasets()),
                    )
                btn_refresh_resume.click(_refresh_resume_lists, None, [resume_lora, resume_extra_ds])

                def _do_resume(lora, ckpt, extra_ds, epochs):
                    tid, msg = resume_training(lora, ckpt, extra_ds, epochs)
                    return msg
                btn_resume.click(
                    _do_resume,
                    [resume_lora, resume_ckpt, resume_extra_ds, resume_epochs],
                    resume_msg,
                )

            gr.Markdown("#### 任务列表")
            table = gr.Dataframe(
                headers=["ID", "名称", "状态", "进度", "信息"],
                datatype=["str", "str", "str", "str", "str"],
                row_count=10, col_count=5, interactive=False,
            )

            def _monitor_wrap(choice):
                """从下拉选项提取 task_id 再刷新"""
                tid = _extract_task_id_from_choice(choice)
                result = monitor_refresh_v2(tid)
                # 同时刷新下拉选项
                choices = _list_task_choices()
                return (*result, gr.update(choices=choices))

            btn_refresh.click(
                _monitor_wrap, watch_id,
                [status, progress_bar, logs, loss_plot, table, watch_id],
            )

            def _cancel_wrap(choice):
                tid = _extract_task_id_from_choice(choice)
                return cancel_task(tid)
            btn_cancel.click(_cancel_wrap, watch_id, status)

            try:
                timer = gr.Timer(3)
                def _auto_refresh(choice, enabled):
                    if not enabled or not (choice or "").strip():
                        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                    tid = _extract_task_id_from_choice(choice)
                    result = monitor_refresh_v2(tid)
                    choices = _list_task_choices()
                    return (*result, gr.update(choices=choices))
                timer.tick(
                    _auto_refresh, [watch_id, auto_refresh],
                    [status, progress_bar, logs, loss_plot, table, watch_id],
                )
            except (AttributeError, Exception):
                pass

            # 切换到任务中心时刷新任务列表
            def _on_monitor_tab_select():
                choices = _list_task_choices()
                resumable = _list_resumable_loras()
                ds_list = _list_datasets()
                return gr.update(choices=choices), gr.update(choices=resumable), gr.update(choices=ds_list)
            tab_monitor.select(_on_monitor_tab_select, None, [watch_id, resume_lora, resume_extra_ds])

            # 点击任务表格行 → 自动选中该任务
            def _on_table_select(evt: gr.SelectData):
                """表格行点击自动选中"""
                if evt.index is not None:
                    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
                    tasks = task_queue.get_all_tasks()
                    if 0 <= row_idx < len(tasks):
                        tid = tasks[row_idx].get("id", "")
                        choices = _list_task_choices()
                        for c in choices:
                            if tid in c:
                                return gr.update(choices=choices, value=c)
                return gr.update()
            table.select(_on_table_select, None, watch_id)

            # 跨 Tab 联动: 任务提交后自动出现在下拉并选中
            def _auto_select_task(task_id_str):
                """其他 Tab 提交任务后，自动更新下拉框选中该任务"""
                if not task_id_str:
                    return gr.update()
                choices = _list_task_choices()
                # 找到匹配的选项
                for c in choices:
                    if task_id_str in c:
                        return gr.update(choices=choices, value=c)
                # 没找到，可能还没刷新到，直接设 task_id 为值
                return gr.update(choices=choices, value=task_id_str)

            # 全部任务 ID 联动（一个不漏）
            for _tid_comp in [train_task_id, merge_task_id, bench_task_id,
                              api_kd_task_id, synth_task_id,
                              me_bake_export_tid, doc_task_id, evo_task_id, mm_task_id,
                              pt_task_id, exp_task_id,
                              hybrid_task_id, gguf_task_id, ollama_task_id]:
                try:
                    _tid_comp.change(_auto_select_task, _tid_comp, watch_id)
                except Exception:
                    pass


        # ================================================================
        #  Tab 7: ⚙️ 设置
        # ================================================================
        with gr.Tab("⚙️ 设置"):
            gr.Markdown("### ForgeX v3.0 — 环境 & 系统设置")

            with gr.Accordion("🖥️ 环境信息", open=True):
                env_info = gr.Markdown(value=_env_report_quick())
                btn_refresh_env = gr.Button("🔄 刷新详细环境信息")
                btn_refresh_env.click(_env_report_md, None, env_info)

            with gr.Accordion("🔧 常用操作", open=True):
                with gr.Row():
                    btn_reload = gr.Button("🔄 重载数据集索引", scale=1)
                    btn_clear_cache = gr.Button("🧹 清空缓存（刷新所有下拉列表）", scale=1)
                    btn_clear_tasks = gr.Button("🗑️ 清理已完成任务", variant="stop", scale=1)
                adv_msg = gr.Textbox(label="状态", interactive=False)
                btn_reload.click(lambda: (dm._load_index() or "✅ 数据集索引已重载"), None, adv_msg)

                def _clear_all_caches():
                    _invalidate_loras_cache()
                    _invalidate_ds_cache()
                    return "✅ 所有缓存已清空，下拉列表将在切换 Tab 时刷新"
                btn_clear_cache.click(_clear_all_caches, None, adv_msg)

                def _clear_done_tasks():
                    tasks = task_queue.get_all_tasks()
                    cleared = 0
                    for t in tasks:
                        if t.get("status") in ("completed", "failed", "cancelled"):
                            tid = t.get("id")
                            if tid and tid in task_queue._tasks:
                                del task_queue._tasks[tid]
                                cleared += 1
                    return f"✅ 已清理 {cleared} 个已完成/失败/取消的任务"
                btn_clear_tasks.click(_clear_done_tasks, None, adv_msg)

            with gr.Accordion("📂 目录信息", open=False):
                def _dir_stats():
                    import os
                    lines = []
                    for name, path in [("模型/LoRA", LORAS_DIR), ("数据集", DATASETS_DIR)]:
                        if path.exists():
                            items = [d for d in path.iterdir() if d.is_dir()]
                            total_size = sum(
                                f.stat().st_size
                                for d in items for f in d.rglob("*") if f.is_file()
                            ) if items else 0
                            lines.append(f"**{name}** (`{path}`): {len(items)} 项 | {total_size/1e9:.1f} GB")
                        else:
                            lines.append(f"**{name}** (`{path}`): 不存在")
                    return "\n\n".join(lines)
                dir_info = gr.Markdown("")
                btn_dir_refresh = gr.Button("📊 统计目录占用")
                btn_dir_refresh.click(_dir_stats, None, dir_info)

        # ════════════════════════════════════════════════════
        #  跨 Tab 联动增强 — 模型编辑器 ↔ 评测/部署/训练
        # ════════════════════════════════════════════════════

        # ── 快捷按钮: 编辑器 → 评测/聊天（同步模型路径） ──
        def _me_goto_eval(model_name):
            if not model_name:
                return gr.update(), "❌ 请先选择模型"
            clean = _extract_model_name_from_label(model_name)
            mp = str(LORAS_DIR / clean)
            return gr.update(value=mp), f"✅ 已同步到聊天测试 — 请切换到 🧪评测 Tab 载入模型"
        btn_me_goto_eval.click(
            _me_goto_eval, [me_model],
            [chat_model_path, me_identity_msg],
        )

        # ── 快捷按钮: 编辑器 → 部署 ──
        def _me_goto_deploy(model_name):
            if not model_name:
                return gr.update(), "❌ 请先选择模型"
            clean = _extract_model_name_from_label(model_name)
            mp = str(LORAS_DIR / clean)
            return gr.update(value=mp), f"✅ 已同步到部署 — 请切换到 🚀部署 Tab"
        btn_me_goto_deploy.click(
            _me_goto_deploy, [me_model],
            [base_for_merge, me_identity_msg],
        )

        # ── 快捷按钮: 编辑器 → 训练（用此模型继续训练）──
        def _me_goto_train(model_name):
            if not model_name:
                return gr.update(), "❌ 请先选择模型"
            return gr.update(value=model_name), f"✅ 已同步到训练 — 请切换到 🔥训练 Tab"
        btn_me_goto_train.click(
            _me_goto_train, [me_model],
            [local_model_picker, me_identity_msg],
        )

        # ── 评测 Tab 选择模型时自动加载系统提示词 ──
        def _eval_sync_system_prompt(model_name):
            """评测 Tab 选模型 → 自动填入编辑好的系统提示词"""
            if not model_name:
                return gr.update()
            try:
                from core.model_editor import load_profile as _lp
                mp = _resolve_model_path(model_name)
                profile = _lp(mp)
                sp = profile.get("system_prompt", "")
                if sp:
                    return gr.update(value=sp)
            except Exception:
                pass
            return gr.update()
        chat_model_picker.change(
            _eval_sync_system_prompt, [chat_model_picker], [chat_sys],
        )

    return demo

app = build_app()


def _launch_app_compat(server_name: str, server_port: int, share: bool):
    """Launch Gradio app without passing removed kwargs on newer Gradio."""
    kwargs = {
        "server_name": server_name,
        "server_port": server_port,
        "share": share,
    }
    if "show_api" in inspect.signature(app.launch).parameters:
        kwargs["show_api"] = False
    return app.launch(**kwargs)


if __name__ == "__main__":
    host = str(config.get("server host", config.get("host", "127.0.0.1")))
    port = int(config.get("server port", config.get("port", 7860)))
    share = bool(config.get("share", False))

    try:
        _launch_app_compat(server_name=host, server_port=port, share=share)
    except ValueError:
        try:
            _launch_app_compat(server_name="127.0.0.1", server_port=port, share=share)
        except ValueError:
            _launch_app_compat(server_name="127.0.0.1", server_port=port, share=True)
