# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

# core/utils.py - ForgeX v2.5 工具集（GPU 感知 + 智能硬件適配）
import json
import subprocess
import threading
import random
import os
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional, Any, Dict, List


def safe_json_load(path: Path, default: Any = None) -> Any:
    try:
        if path.exists() and path.stat().st_size > 0:
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default if default is not None else {}


def safe_json_save(path: Path, data: Any) -> bool:
    """Best-effort JSON save.

    - Never crash callers: return False on failure.
    - Make common non-JSON objects serializable (Path, numpy scalars, datasets LazyRow, etc.)
      by coercing mappings to dict and falling back to str().
    - Protected against RecursionError from circular/deeply-nested objects.
    """
    _seen = set()  # 防循环引用追踪

    def _default(o):
        oid = id(o)
        if oid in _seen:
            return "<circular ref>"
        _seen.add(oid)
        try:
            # datasets LazyRow / mapping-like → 但不要展开 Gradio 组件等复杂对象
            if (hasattr(o, "keys") and callable(getattr(o, "keys"))
                    and hasattr(o, "__getitem__")
                    and not hasattr(o, "render")):  # 排除 Gradio 组件
                try:
                    return {str(k): o[k] for k in list(o.keys())[:50]}
                except Exception:
                    pass
        except Exception:
            pass
        try:
            s = str(o)
            return s[:500] if len(s) > 500 else s
        except Exception:
            return repr(type(o))

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_default), encoding="utf-8")
        tmp.replace(path)
        return True
    except (RecursionError, ValueError):
        # 核弹后备: 逐顶层键序列化
        try:
            if isinstance(data, dict):
                safe = {}
                for k, v in data.items():
                    try:
                        json.dumps(v, default=str)
                        safe[k] = v
                    except Exception:
                        safe[k] = str(v)[:500]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(safe, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
                return True
            else:
                path.write_text(str(data)[:10000], encoding="utf-8")
                return True
        except Exception:
            return False
    except Exception:
        return False


def human_size(size_bytes: int) -> str:
    if size_bytes < 0:
        return "N/A"
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {u}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def random_name(prefix="forged"):
    return f"{prefix}_{get_timestamp()}_{random.randint(1000, 9999)}"


def run_subprocess(cmd, cwd=None, progress_cb=None, env=None):
    from core.logger import log
    log(f"CMD: {' '.join(str(c) for c in cmd)}")
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=run_env,
    )
    output_lines = []
    for line in proc.stdout:
        line = line.strip()
        if line:
            log(line)
            output_lines.append(line)
            if progress_cb:
                progress_cb(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (rc={proc.returncode}): {' '.join(output_lines[-5:])}")
    return proc.returncode


def run_in_thread(func):
    def wrapper(*a, **kw):
        t = threading.Thread(target=func, args=a, kwargs=kw, daemon=True)
        t.start()
        return t
    return wrapper


# ===================== GPU 硬件數據庫（精確到型號）=====================
# (vram_mb, compute_capability, supports_bf16, supports_flash_attn2, recommended_batch, recommended_rank, max_seq_for_7b_qlora)
GPU_DATABASE = {
    # === RTX 50 系列（Blackwell）===
    "5090":      (32768, 12.0, True,  True,  8,  128, 4096),
    "5080":      (16384, 12.0, True,  True,  8,  128, 4096),
    "5070 ti":   (16384, 12.0, True,  True,  8,  128, 4096),
    "5070":      (12288, 12.0, True,  True,  4,  64,  2048),
    "5060 ti 16":(16384, 12.0, True,  True,  8,  128, 4096),
    "5060 ti":   (8192,  12.0, True,  True,  2,  64,  2048),
    "5060":      (8192,  12.0, True,  True,  2,  64,  2048),
    # === RTX 40 系列（Ada Lovelace）===
    "4090":      (24576, 8.9,  True,  True,  8,  128, 4096),
    "4080 super":(16384, 8.9,  True,  True,  8,  128, 4096),
    "4080":      (16384, 8.9,  True,  True,  8,  128, 4096),
    "4070 ti super":(16384, 8.9, True, True, 8,  128, 4096),
    "4070 ti":   (12288, 8.9,  True,  True,  4,  64,  2048),
    "4070 super":(12288, 8.9,  True,  True,  4,  64,  2048),
    "4070":      (12288, 8.9,  True,  True,  4,  64,  2048),
    "4060 ti 16":(16384, 8.9,  True,  True,  4,  64,  2048),
    "4060 ti":   (8192,  8.9,  True,  True,  2,  32,  2048),
    "4060":      (8192,  8.9,  True,  True,  2,  32,  2048),
    # === RTX 30 系列（Ampere）===
    "3090 ti":   (24576, 8.6,  True,  True,  8,  128, 4096),
    "3090":      (24576, 8.6,  True,  True,  8,  128, 4096),
    "3080 ti":   (12288, 8.6,  True,  True,  4,  64,  2048),
    "3080":      (10240, 8.6,  True,  True,  4,  64,  2048),
    "3070 ti":   (8192,  8.6,  True,  True,  2,  32,  2048),
    "3070":      (8192,  8.6,  True,  True,  2,  32,  2048),
    "3060 ti":   (8192,  8.6,  True,  True,  2,  32,  2048),
    "3060":      (12288, 8.6,  True,  True,  4,  64,  2048),
    # === RTX 20 系列（Turing）===
    "2080 ti":   (11264, 7.5,  False, False, 2,  32,  1024),
    "2080 super":(8192,  7.5,  False, False, 2,  32,  1024),
    "2080":      (8192,  7.5,  False, False, 2,  32,  1024),
    "2070":      (8192,  7.5,  False, False, 2,  32,  1024),
    "2060":      (6144,  7.5,  False, False, 1,  16,  512),
    # === 數據中心 ===
    "a100 80":   (81920, 8.0,  True,  True,  16, 256, 8192),
    "a100 40":   (40960, 8.0,  True,  True,  16, 256, 8192),
    "a100":      (81920, 8.0,  True,  True,  16, 256, 8192),
    "a6000":     (49152, 8.6,  True,  True,  16, 256, 8192),
    "l40s":      (49152, 8.9,  True,  True,  16, 256, 8192),
    "l40":       (49152, 8.9,  True,  True,  16, 256, 8192),
    "l4":        (24576, 8.9,  True,  True,  8,  128, 4096),
    "h100":      (81920, 9.0,  True,  True,  16, 256, 8192),
    "t4":        (16384, 7.5,  False, False, 4,  32,  2048),
}


def detect_gpu() -> Dict:
    """精確 GPU 檢測 + 能力分析"""
    info = {
        "name": "unknown", "vram_mb": 0, "count": 0,
        "compute_capability": 0.0, "supports_bf16": False,
        "supports_flash_attn": False,
        "recommended_batch": 2, "recommended_rank": 32,
        "max_seq_for_7b": 2048,
        "driver_version": "", "cuda_version": "",
    }
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        ).strip()
        gpus = [line.strip().split(", ") for line in out.split("\n") if line.strip()]
        if gpus:
            info["count"] = len(gpus)
            info["name"] = gpus[0][0]
            info["vram_mb"] = int(float(gpus[0][1]))
            if len(gpus[0]) > 2:
                info["driver_version"] = gpus[0][2]

            # 精確匹配 GPU 型號（從長到短匹配）
            name_lower = info["name"].lower()
            matched = False
            for pattern in sorted(GPU_DATABASE.keys(), key=len, reverse=True):
                if pattern in name_lower:
                    vram, cc, bf16, fa2, batch, rank, max_seq = GPU_DATABASE[pattern]
                    info["compute_capability"] = cc
                    info["supports_bf16"] = bf16
                    info["supports_flash_attn"] = fa2
                    info["recommended_batch"] = batch
                    info["recommended_rank"] = rank
                    info["max_seq_for_7b"] = max_seq
                    matched = True
                    break

            if not matched:
                # 未知 GPU，根據 VRAM 猜測
                vram = info["vram_mb"]
                if vram >= 24000:
                    info.update(recommended_batch=8, recommended_rank=128, max_seq_for_7b=4096)
                elif vram >= 12000:
                    info.update(recommended_batch=4, recommended_rank=64, max_seq_for_7b=2048)
                elif vram >= 8000:
                    info.update(recommended_batch=2, recommended_rank=32, max_seq_for_7b=2048)
                else:
                    info.update(recommended_batch=1, recommended_rank=16, max_seq_for_7b=512)
                # 嘗試通過 torch 獲取 compute capability
                try:
                    import torch
                    if torch.cuda.is_available():
                        cc = torch.cuda.get_device_capability(0)
                        info["compute_capability"] = float(f"{cc[0]}.{cc[1]}")
                        info["supports_bf16"] = cc[0] >= 8
                        info["supports_flash_attn"] = cc[0] >= 8
                except Exception:
                    pass

            # CUDA 版本
            try:
                import torch
                info["cuda_version"] = torch.version.cuda or ""
            except Exception:
                pass

    except Exception:
        pass
    return info


def get_gpu_status() -> Dict:
    """獲取 GPU 即時狀態（使用率、溫度、VRAM 使用）"""
    status = {"gpu_util": 0, "mem_used_mb": 0, "mem_total_mb": 0, "temperature": 0, "power_w": 0}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        ).strip()
        parts = out.split("\n")[0].split(", ")
        if len(parts) >= 5:
            status["gpu_util"] = int(float(parts[0]))
            status["mem_used_mb"] = int(float(parts[1]))
            status["mem_total_mb"] = int(float(parts[2]))
            status["temperature"] = int(float(parts[3]))
            status["power_w"] = round(float(parts[4]), 1)
    except Exception:
        pass
    return status


# ===================== 訓練精度自動選擇 =====================
def auto_dtype(gpu_info: Dict) -> str:
    """根據 GPU 能力自動選擇最優精度"""
    if gpu_info.get("supports_bf16"):
        return "bf16"
    return "fp16"


def check_flash_attention() -> bool:
    """檢查 Flash Attention 2 是否可用"""
    try:
        import flash_attn
        return True
    except ImportError:
        return False


# ===================== LoRA Target 自動檢測 =====================
# 不同架構的最佳 LoRA target modules
LORA_TARGETS_BY_ARCH = {
    "qwen2":    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "llama":    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral":  ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma":    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi":      ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
    "chatglm":  ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "baichuan": ["W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "internlm": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "yi":       ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "deepseek": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "default":  ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def detect_model_arch(model_name: str) -> str:
    """根據模型名稱檢測架構"""
    name_lower = model_name.lower()
    for arch in LORA_TARGETS_BY_ARCH:
        if arch != "default" and arch in name_lower:
            return arch
    return "default"


def get_lora_targets(model_name: str) -> List[str]:
    """獲取模型的最佳 LoRA target modules"""
    arch = detect_model_arch(model_name)
    return LORA_TARGETS_BY_ARCH.get(arch, LORA_TARGETS_BY_ARCH["default"])


# ===================== VRAM 估算（精確版）=====================
def estimate_vram_mb(model_params_b: float, method: str = "qlora", batch_size: int = 4,
                     seq_length: int = 2048, rank: int = 64,
                     gradient_checkpointing: bool = True) -> Dict:
    """精確 VRAM 估算（考慮梯度檢查點、優化器狀態）"""
    if method == "qlora":
        model_mb = model_params_b * 1e9 * 0.5 / (1024 ** 2)  # 4-bit ≈ 0.5 bytes/param
    elif method == "lora":
        model_mb = model_params_b * 1e9 * 2 / (1024 ** 2)    # fp16 = 2 bytes/param
    else:  # full
        model_mb = model_params_b * 1e9 * 2 / (1024 ** 2)

    # LoRA 參數量（估算）
    # 典型 7B 模型有 ~32 層，每層 q/k/v/o/gate/up/down = 7 個 target
    # 每個 target 的 LoRA 參數 = hidden_size * rank * 2
    hidden_size = int(model_params_b * 570)  # 粗估
    num_targets = 7  # 典型全 target
    num_layers = max(1, int(model_params_b * 4.5))
    lora_params = num_layers * num_targets * hidden_size * rank * 2
    lora_mb = lora_params * 2 / (1024 ** 2)  # fp16

    # Activation memory
    if gradient_checkpointing:
        # 梯度檢查點大幅減少激活內存（約到 sqrt(layers)）
        act_mb = batch_size * seq_length * hidden_size * 2 * (num_layers ** 0.5) / (1024 ** 2)
    else:
        act_mb = batch_size * seq_length * hidden_size * 2 * num_layers * 0.3 / (1024 ** 2)

    # Gradient memory（LoRA only）
    grad_mb = lora_mb * 1.0 if method != "full" else model_mb * 1.0

    # Optimizer memory（AdamW 8-bit: ~2x params, standard: ~8x）
    if method in ("qlora", "lora"):
        opt_mb = lora_mb * 4  # 8-bit AdamW 約 4 倍 LoRA 參數
    else:
        opt_mb = model_mb * 8

    total = model_mb + lora_mb + act_mb + grad_mb + opt_mb
    overhead = total * 0.1  # CUDA 開銷

    return {
        "model_mb": round(model_mb), "lora_mb": round(lora_mb),
        "activation_mb": round(act_mb), "gradient_mb": round(grad_mb),
        "optimizer_mb": round(opt_mb), "overhead_mb": round(overhead),
        "total_mb": round(total + overhead),
        "total_gb": round((total + overhead) / 1024, 1),
        "grad_ckpt": gradient_checkpointing,
    }


MODEL_SIZE_MAP = {
    "0.5b": 0.5, "1b": 1, "1.5b": 1.5, "3b": 3, "7b": 7, "8b": 8,
    "13b": 13, "14b": 14, "32b": 32, "70b": 70, "72b": 72,
}


def guess_model_size(name: str) -> float:
    nl = name.lower()
    for k, s in sorted(MODEL_SIZE_MAP.items(), key=lambda x: -x[1]):
        if k in nl:
            return s
    return 7.0


def auto_train_params(gpu_info: Dict, model_size_b: float, method: str = "qlora") -> Dict:
    """智能訓練參數推薦（RTX 5060/4060 重點優化）"""
    vram = gpu_info.get("vram_mb", 8192)
    available = vram * 0.90  # 留 10% 系統開銷

    # 先用最小配置估算
    est_min = estimate_vram_mb(model_size_b, method, 1, 1024, 16, True)

    if est_min["total_mb"] > available:
        return {
            "batch_size": 1, "rank": 8, "alpha": 16, "max_seq_length": 512,
            "gradient_accumulation_steps": 32, "gradient_checkpointing": True,
            "warning": f"⚠️ VRAM 不足 ({vram}MB)，模型可能太大。建議使用更小的模型或開啟 QLoRA。",
            "dtype": auto_dtype(gpu_info),
        }

    # 逐步增大配置直到接近上限
    configs = []
    for rank in [16, 32, 64, 128]:
        for batch in [1, 2, 4, 8]:
            for seq in [1024, 2048, 4096]:
                est = estimate_vram_mb(model_size_b, method, batch, seq, rank, True)
                if est["total_mb"] < available * 0.85:
                    # 有效批次 = batch * grad_accum，目標 16-32
                    ga = max(1, min(32, 16 // batch))
                    configs.append({
                        "batch_size": batch, "rank": rank, "alpha": rank * 2,
                        "max_seq_length": seq,
                        "gradient_accumulation_steps": ga,
                        "gradient_checkpointing": True,
                        "vram_est": est["total_gb"],
                        "effective_batch": batch * ga,
                    })

    if not configs:
        return {
            "batch_size": 1, "rank": 16, "alpha": 32, "max_seq_length": 1024,
            "gradient_accumulation_steps": 16, "gradient_checkpointing": True,
            "warning": f"VRAM 緊張 ({vram}MB)，已使用最小配置",
            "dtype": auto_dtype(gpu_info),
        }

    # 選最優配置：優先大 effective_batch → 大 rank → 大 seq
    best = max(configs, key=lambda c: (c["effective_batch"], c["rank"], c["max_seq_length"]))
    best["warning"] = None
    best["dtype"] = auto_dtype(gpu_info)
    return best

# Alias for backward compatibility
detect_target_modules = get_lora_targets

