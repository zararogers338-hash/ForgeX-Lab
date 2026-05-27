# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

"""ForgeX 智能参数推荐 — 根据模型/数据/硬件动态计算最佳训练参数。

策略:
  - 模型参数量 → rank、学习率基准
  - 数据条数 → epochs、batch 策略
  - 序列长度 → 显存预估、batch 上限
  - GPU VRAM → QLoRA 决策、batch 大小
  - 数据质量 → 调整正则化参数
"""
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

from core import DATASETS_DIR, log


# ════════════════════════════════════════════
#  模型规格查询
# ════════════════════════════════════════════

# 常见模型的参数量（B），用于自动推导
_MODEL_SIZE_DB = {
    "0.5b": 0.5, "0.5": 0.5,
    "1b": 1.0, "1.5b": 1.5,
    "2b": 2.0, "3b": 3.0, "3.8b": 3.8,
    "7b": 7.0, "8b": 8.0,
    "14b": 14.0, "13b": 13.0,
    "32b": 32.0, "70b": 70.0,
}


def _guess_model_size_b(model_id: str) -> float:
    """从模型 ID 猜测参数量（十亿）"""
    model_id_lower = model_id.lower().replace("-", "").replace("_", "")
    # 尝试直接匹配常见模式
    import re
    # "1.5B", "7B", "0.5b" 等
    m = re.search(r'(\d+\.?\d*)\s*b(?:ase|illion|\b|$|-)', model_id_lower)
    if m:
        val = float(m.group(1))
        if val < 200:  # 合理范围
            return val
    # Phi-3.5-mini → 3.8B
    if "phi" in model_id_lower and "mini" in model_id_lower:
        return 3.8
    # Gemma-2-2b
    for size_key, size_val in _MODEL_SIZE_DB.items():
        if size_key in model_id_lower:
            return size_val
    return 0  # 未知


def _count_dataset_rows(ds_names) -> int:
    """估算数据集总行数"""
    if isinstance(ds_names, str):
        ds_names = [ds_names]
    total = 0
    for name in (ds_names or []):
        p = Path(DATASETS_DIR) / name
        if not p.exists():
            continue
        try:
            if p.is_dir():
                for f in list(p.glob("*.json")) + list(p.glob("*.jsonl")):
                    if f.suffix == ".jsonl":
                        total += sum(1 for line in f.read_text(encoding="utf-8").strip().splitlines() if line.strip())
                    else:
                        data = json.loads(f.read_text(encoding="utf-8"))
                        total += len(data) if isinstance(data, list) else 1
            elif p.is_file():
                text = p.read_text(encoding="utf-8")
                if p.suffix == ".jsonl":
                    total += sum(1 for line in text.strip().splitlines() if line.strip())
                elif p.suffix == ".json":
                    data = json.loads(text)
                    total += len(data) if isinstance(data, list) else 1
                else:
                    total += sum(1 for line in text.strip().splitlines() if line.strip())
        except Exception:
            pass
    return total


# ════════════════════════════════════════════
#  VRAM 估算
# ════════════════════════════════════════════

def estimate_vram_mb(model_size_b: float, seq_len: int, batch_size: int,
                     rank: int, use_qlora: bool, ga_steps: int = 4) -> Dict[str, Any]:
    """估算训练显存需求（MB）

    公式（经验近似）:
    - 模型权重: params_B × (2 if fp16 else 4) × (0.5 if qlora else 1) GB
    - LoRA 权重: rank × hidden × 2 × layers × 2 bytes (很小)
    - 优化器: LoRA参数 × 8 bytes（AdamW 两个 moment）× (0.25 if 8bit else 1)
    - 激活内存: batch × seq × hidden × layers × 2 bytes / (checkpointing_factor)
    - KV Cache: batch × seq × hidden × layers × 4 bytes
    """
    if model_size_b <= 0:
        return {"total_mb": 0, "breakdown": {}, "error": "未知模型大小"}

    # 估算 hidden_dim 和 layers
    if model_size_b <= 1:
        hidden, layers = 2048, 24
    elif model_size_b <= 3:
        hidden, layers = 2560, 32
    elif model_size_b <= 4:
        hidden, layers = 3072, 32
    elif model_size_b <= 8:
        hidden, layers = 4096, 32
    elif model_size_b <= 14:
        hidden, layers = 5120, 40
    elif model_size_b <= 32:
        hidden, layers = 6656, 60
    else:
        hidden, layers = 8192, 80

    # 模型权重
    weight_mb = model_size_b * 1024  # fp16 = 2B/param
    if use_qlora:
        weight_mb *= 0.5  # 4bit ≈ 半

    # LoRA 参数 (rank × hidden × 2 matrices × layers × adapter_count)
    lora_params = rank * hidden * 2 * layers * 2  # q,v
    lora_mb = lora_params * 2 / 1024 / 1024  # fp16

    # 优化器状态（8bit paged = 1/4 of fp32 AdamW）
    optim_mb = lora_params * 8 / 1024 / 1024 * 0.25  # 8bit

    # 激活内存（带 gradient checkpointing）
    # 粗略: batch × seq × hidden × layers × 2 bytes / 3 (checkpoint factor)
    activation_mb = (batch_size * seq_len * hidden * layers * 2) / 1024 / 1024 / 3

    # KV Cache
    kv_mb = (batch_size * seq_len * hidden * layers * 4) / 1024 / 1024

    total = weight_mb + lora_mb + optim_mb + activation_mb + kv_mb
    # 加 15% 碎片/overhead
    total *= 1.15

    return {
        "total_mb": int(total),
        "total_gb": round(total / 1024, 1),
        "breakdown": {
            "模型权重": f"{weight_mb:.0f} MB",
            "LoRA 参数": f"{lora_mb:.0f} MB",
            "优化器状态": f"{optim_mb:.0f} MB",
            "激活内存": f"{activation_mb:.0f} MB",
            "KV Cache": f"{kv_mb:.0f} MB",
        },
    }


# ════════════════════════════════════════════
#  智能推荐
# ════════════════════════════════════════════

def recommend_params(model_id: str, ds_names=None,
                     vram_mb: int = 0) -> Dict[str, Any]:
    """根据模型/数据/硬件推荐训练参数

    Returns:
        {
            "lr": float,
            "epochs": float,
            "batch_size": int,
            "rank": int,
            "max_seq_len": int,
            "gradient_accumulation_steps": int,
            "use_qlora": bool,
            "warmup_ratio": float,
            "reasoning": list[str],  # 每条决策的理由
            "estimated_time_min": int,
            "estimated_vram_gb": float,
        }
    """
    model_size = _guess_model_size_b(model_id)
    n_rows = _count_dataset_rows(ds_names) if ds_names else 0
    vram_gb = vram_mb / 1024 if vram_mb > 0 else 0

    reasoning = []
    rec: Dict[str, Any] = {}

    # ─── 模型大小 → rank, lr 基准 ───
    if model_size <= 0:
        rec["rank"] = 64
        rec["lr"] = 2e-4
        reasoning.append(f"⚙️ 未识别模型大小，使用默认 rank=64, lr=2e-4")
    elif model_size <= 1:
        rec["rank"] = 32
        rec["lr"] = 3e-4
        reasoning.append(f"📐 小模型 ({model_size}B) → rank=32 足够, lr=3e-4 (小模型可用较高学习率)")
    elif model_size <= 3:
        rec["rank"] = 48
        rec["lr"] = 2e-4
        reasoning.append(f"📐 中小模型 ({model_size}B) → rank=48, lr=2e-4")
    elif model_size <= 8:
        rec["rank"] = 64
        rec["lr"] = 2e-4
        reasoning.append(f"📐 中模型 ({model_size}B) → rank=64, lr=2e-4 (标准配置)")
    elif model_size <= 14:
        rec["rank"] = 64
        rec["lr"] = 1e-4
        reasoning.append(f"📐 大模型 ({model_size}B) → rank=64, lr=1e-4 (大模型需要更低学习率)")
    else:
        rec["rank"] = 32
        rec["lr"] = 5e-5
        reasoning.append(f"📐 超大模型 ({model_size}B) → rank=32 省显存, lr=5e-5")

    # ─── 数据量 → epochs ───
    if n_rows <= 0:
        rec["epochs"] = 3.0
        reasoning.append("📊 未检测到数据，默认 3 epochs")
    elif n_rows < 100:
        rec["epochs"] = 5.0
        reasoning.append(f"📊 极少数据 ({n_rows}条) → 5 epochs (多看几遍)")
    elif n_rows < 500:
        rec["epochs"] = 3.0
        reasoning.append(f"📊 少量数据 ({n_rows}条) → 3 epochs")
    elif n_rows < 2000:
        rec["epochs"] = 2.0
        reasoning.append(f"📊 中等数据 ({n_rows}条) → 2 epochs")
    elif n_rows < 10000:
        rec["epochs"] = 1.0
        reasoning.append(f"📊 较多数据 ({n_rows}条) → 1 epoch")
    else:
        rec["epochs"] = 0.5
        reasoning.append(f"📊 大量数据 ({n_rows}条) → 0.5 epoch (数据量充足)")

    # ─── VRAM → batch, qlora, seq_len ───
    if vram_gb <= 0:
        rec["batch_size"] = 1
        rec["gradient_accumulation_steps"] = 4
        rec["use_qlora"] = False
        rec["max_seq_len"] = 2048
        reasoning.append("💾 未检测到 GPU，使用保守参数")
    elif vram_gb < 6:
        rec["batch_size"] = 1
        rec["gradient_accumulation_steps"] = 8
        rec["use_qlora"] = True
        rec["max_seq_len"] = 1024
        reasoning.append(f"💾 低显存 ({vram_gb:.0f}GB) → QLoRA 必须开, batch=1, seq=1024")
    elif vram_gb < 10:
        rec["batch_size"] = 1
        rec["gradient_accumulation_steps"] = 4
        rec["use_qlora"] = model_size > 3
        rec["max_seq_len"] = 2048
        qlora_note = "开启QLoRA" if rec["use_qlora"] else "无需QLoRA"
        reasoning.append(f"💾 中等显存 ({vram_gb:.0f}GB) → {qlora_note}, batch=1, seq=2048")
    elif vram_gb < 16:
        rec["batch_size"] = 2
        rec["gradient_accumulation_steps"] = 4
        rec["use_qlora"] = model_size > 8
        rec["max_seq_len"] = 2048
        reasoning.append(f"💾 较好显存 ({vram_gb:.0f}GB) → batch=2, seq=2048")
    else:
        rec["batch_size"] = 4
        rec["gradient_accumulation_steps"] = 2
        rec["use_qlora"] = model_size > 14
        rec["max_seq_len"] = 4096
        reasoning.append(f"💾 充裕显存 ({vram_gb:.0f}GB) → batch=4, seq=4096")

    # ─── 预热 ───
    rec["warmup_ratio"] = 0.05 if n_rows > 500 else 0.1
    if n_rows > 0 and n_rows < 200:
        reasoning.append("🌡️ 小数据集 → warmup=0.1 (更平滑起步)")

    # ─── 估算训练时间 ───
    if n_rows > 0 and model_size > 0:
        eff_batch = rec["batch_size"] * rec["gradient_accumulation_steps"]
        steps = int(n_rows * rec["epochs"] / eff_batch)
        # 粗略: 小模型 ~2s/step, 大模型 ~5s/step
        sec_per_step = 1.5 + model_size * 0.4
        if rec["use_qlora"]:
            sec_per_step *= 1.3  # QLoRA 略慢
        est_min = int(steps * sec_per_step / 60)
        rec["estimated_time_min"] = max(1, est_min)
        reasoning.append(f"⏱️ 预计 ~{steps} steps, 约 {est_min} 分钟")
    else:
        rec["estimated_time_min"] = 0

    # ─── VRAM 估算 ───
    if model_size > 0:
        vram_est = estimate_vram_mb(
            model_size, rec["max_seq_len"], rec["batch_size"],
            rec["rank"], rec["use_qlora"], rec["gradient_accumulation_steps"])
        rec["estimated_vram_gb"] = vram_est["total_gb"]
        rec["vram_breakdown"] = vram_est["breakdown"]

        # 如果预估超过实际 VRAM，自动降级
        if vram_gb > 0 and vram_est["total_gb"] > vram_gb * 0.9:
            reasoning.append(f"⚠️ 预估显存 {vram_est['total_gb']}GB > 可用 {vram_gb:.0f}GB，自动降级")
            if not rec["use_qlora"]:
                rec["use_qlora"] = True
                reasoning.append("  → 开启 QLoRA")
            if rec["batch_size"] > 1:
                rec["batch_size"] = 1
                rec["gradient_accumulation_steps"] = max(rec["gradient_accumulation_steps"], 8)
                reasoning.append("  → batch=1, 增加梯度累积")
            if rec["max_seq_len"] > 1024:
                rec["max_seq_len"] = 1024
                reasoning.append("  → 序列长度降至 1024")

    rec["reasoning"] = reasoning
    return rec


def format_recommendation_markdown(rec: Dict) -> str:
    """格式化推荐参数为简洁的 Markdown"""
    lines = []
    lines.append("### 🎯 智能推荐参数")
    lines.append("")

    # 核心参数
    qlora = "✅" if rec.get("use_qlora") else "❌"
    lines.append(
        f"**lr**={rec.get('lr', '?')} | **rank**={rec.get('rank', '?')} | "
        f"**epochs**={rec.get('epochs', '?')} | **batch**={rec.get('batch_size', '?')} | "
        f"**seq**={rec.get('max_seq_len', '?')} | **QLoRA**={qlora}")

    # 估算
    est_time = rec.get("estimated_time_min", 0)
    est_vram = rec.get("estimated_vram_gb", 0)
    if est_time or est_vram:
        parts = []
        if est_time:
            parts.append(f"⏱️ ~{est_time}分钟")
        if est_vram:
            parts.append(f"💾 ~{est_vram}GB")
        lines.append(f"预估: {' | '.join(parts)}")

    # 理由
    lines.append("")
    for r in rec.get("reasoning", []):
        lines.append(f"  {r}")

    return "\n".join(lines)
