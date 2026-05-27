# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

"""ForgeX 知识浓缩器 (Knowledge Condenser)

目标: 让 1B 模型达到 10B 的效果。

核心理念 — "锻压循环" (Forge-Press Cycle):
  膨胀 → 灌注 → 锻压 → 自蒸馏 → 困难挖掘 → 再灌注 → 循环

每轮循环提高知识密度（每参数承载的信息量），反复淬火后模型变得
小而精，像折叠锻打的大马士革钢 — 层层叠叠，每一层都有用。

主要能力:
  1. 重要性评分 — 量化每个 head/neuron/layer 的贡献
  2. 结构化剪枝 — 砍掉摸鱼的 head、neuron、甚至整层
  3. 自蒸馏     — 大模型教自己的精简版（软标签 + 中间层对齐）
  4. 困难样本挖掘 — 找出模型薄弱环节，集中再训练
  5. 锻压循环   — 全自动 expand → train → prune → distill → mine → retrain
"""

from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core import LORAS_DIR, DATASETS_DIR, log


def _safe_update(task, p, msg):
    if task is not None:
        try:
            task.update_progress(float(p), str(msg))
        except Exception:
            pass


# ════════════════════════════════════════════════════════════
#  1. 重要性评分 (Importance Scoring)
# ════════════════════════════════════════════════════════════

def score_importance(
    model_path: str,
    calibration_data_path: str,
    method: str = "combined",
    n_samples: int = 128,
    task=None,
) -> Dict[str, Any]:
    """量化模型每个结构的重要性

    评分维度:
      - head_scores[layer][head]:  注意力头贡献度
      - neuron_scores[layer][idx]: MLP 神经元贡献度
      - layer_scores[layer]:       整层贡献度

    方法:
      - "magnitude":  权重绝对值（快但粗）
      - "gradient":   梯度敏感度（用校准数据计算 loss 对参数的梯度）
      - "activation": 激活值统计（衡量实际使用频率）
      - "combined":   三者加权融合（最准，默认）

    Returns: {head_scores, neuron_scores, layer_scores, summary}
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _safe_update(task, 2, f"加载模型: {model_path}")
    # 预导入模型类（防止 LazyAutoMapping 失效）
    try:
        from core.safe_loader import ensure_model_importable
        ensure_model_importable(model_path)
    except Exception:
        pass
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32,
        trust_remote_code=True, device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()

    config = model.config
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    hidden = config.hidden_size
    head_dim = hidden // n_heads
    intermediate = getattr(config, "intermediate_size", hidden * 4)

    # 准备校准数据
    _safe_update(task, 8, "准备校准数据...")
    cal_texts = _load_calibration_texts(calibration_data_path, n_samples)

    # ── Magnitude scores ──
    _safe_update(task, 12, "计算权重幅度分数...")
    head_mag, neuron_mag, layer_mag = _magnitude_scores(model, n_layers, n_heads, head_dim, intermediate)

    # ── Gradient scores (需要校准数据) ──
    head_grad = [[0.0] * n_heads for _ in range(n_layers)]
    neuron_grad = [[0.0] * intermediate for _ in range(n_layers)]
    layer_grad = [0.0] * n_layers

    if method in ("gradient", "combined") and cal_texts:
        _safe_update(task, 25, "计算梯度敏感度...")
        head_grad, neuron_grad, layer_grad = _gradient_scores(
            model, tokenizer, cal_texts, n_layers, n_heads, head_dim, intermediate,
            task=task, progress_base=25, progress_range=35,
        )

    # ── Activation scores ──
    head_act = [[0.0] * n_heads for _ in range(n_layers)]
    neuron_act = [[0.0] * intermediate for _ in range(n_layers)]
    layer_act = [0.0] * n_layers

    if method in ("activation", "combined") and cal_texts:
        _safe_update(task, 62, "计算激活统计...")
        head_act, neuron_act, layer_act = _activation_scores(
            model, tokenizer, cal_texts, n_layers, n_heads, head_dim, intermediate,
            task=task, progress_base=62, progress_range=20,
        )

    # ── 融合 ──
    _safe_update(task, 85, "融合分数...")
    if method == "magnitude":
        w_mag, w_grad, w_act = 1.0, 0.0, 0.0
    elif method == "gradient":
        w_mag, w_grad, w_act = 0.0, 1.0, 0.0
    elif method == "activation":
        w_mag, w_grad, w_act = 0.0, 0.0, 1.0
    else:  # combined
        w_mag, w_grad, w_act = 0.3, 0.5, 0.2

    head_scores = []
    neuron_scores = []
    layer_scores = []

    for l in range(n_layers):
        h_scores = []
        for h in range(n_heads):
            s = w_mag * _norm(head_mag[l][h], head_mag) + \
                w_grad * _norm(head_grad[l][h], head_grad) + \
                w_act * _norm(head_act[l][h], head_act)
            h_scores.append(s)
        head_scores.append(h_scores)

        n_scores = []
        for ni in range(min(intermediate, len(neuron_mag[l]))):
            s = w_mag * _norm(neuron_mag[l][ni], neuron_mag) + \
                w_grad * _norm(neuron_grad[l][ni], neuron_grad) + \
                w_act * _norm(neuron_act[l][ni], neuron_act)
            n_scores.append(s)
        neuron_scores.append(n_scores)

        ls = w_mag * _norm(layer_mag[l], [layer_mag]) + \
             w_grad * _norm(layer_grad[l], [layer_grad]) + \
             w_act * _norm(layer_act[l], [layer_act])
        layer_scores.append(ls)

    # 统计
    all_head = [s for layer in head_scores for s in layer]
    all_neuron = [s for layer in neuron_scores for s in layer]

    summary = {
        "n_layers": n_layers, "n_heads": n_heads,
        "heads_below_10pct": sum(1 for s in all_head if s < 0.1),
        "heads_below_25pct": sum(1 for s in all_head if s < 0.25),
        "neurons_below_10pct": sum(1 for s in all_neuron if s < 0.1),
        "weakest_layers": sorted(range(n_layers), key=lambda i: layer_scores[i])[:3],
        "strongest_layers": sorted(range(n_layers), key=lambda i: layer_scores[i])[-3:],
    }

    result = {
        "head_scores": head_scores,
        "neuron_scores": neuron_scores,
        "layer_scores": layer_scores,
        "summary": summary,
        "model_path": model_path,
        "method": method,
    }

    del model
    gc.collect()
    _safe_update(task, 100, f"✅ 重要性评分完成 | {summary['heads_below_25pct']}/{len(all_head)} 头可剪")
    return result


def _magnitude_scores(model, n_layers, n_heads, head_dim, intermediate):
    """基于权重绝对值的重要性"""
    import torch

    head_scores = [[0.0] * n_heads for _ in range(n_layers)]
    neuron_scores = [[0.0] * intermediate for _ in range(n_layers)]
    layer_scores = [0.0] * n_layers

    sd = model.state_dict()
    for key, tensor in sd.items():
        k = key.lower()
        # 找层号
        layer_idx = _extract_layer_idx(k)
        if layer_idx is None or layer_idx >= n_layers:
            continue

        t_abs = tensor.float().abs()

        if "q_proj" in k and "weight" in k:
            # [n_heads * head_dim, hidden] → 每个 head 的幅度
            for h in range(n_heads):
                start = h * head_dim
                end = start + head_dim
                if end <= t_abs.shape[0]:
                    head_scores[layer_idx][h] += t_abs[start:end].mean().item()

        elif ("gate_proj" in k or "up_proj" in k) and "weight" in k:
            for ni in range(min(intermediate, t_abs.shape[0])):
                neuron_scores[layer_idx][ni] += t_abs[ni].mean().item()

        # 全层
        layer_scores[layer_idx] += t_abs.mean().item()

    return head_scores, neuron_scores, layer_scores


def _gradient_scores(model, tokenizer, texts, n_layers, n_heads, head_dim, intermediate,
                     task=None, progress_base=25, progress_range=35):
    """基于梯度敏感度的重要性: loss 对参数的偏导"""
    import torch

    head_scores = [[0.0] * n_heads for _ in range(n_layers)]
    neuron_scores = [[0.0] * intermediate for _ in range(n_layers)]
    layer_scores = [0.0] * n_layers

    model.train()  # 需要梯度
    n = len(texts)

    for ti, text in enumerate(texts):
        if ti % 10 == 0:
            _safe_update(task, progress_base + progress_range * ti / n,
                         f"梯度分析 {ti}/{n}")

        tokens = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=512, padding=False)
        input_ids = tokens["input_ids"]
        if input_ids.shape[1] < 2:
            continue

        model.zero_grad()
        try:
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            if loss is not None:
                loss.backward()
        except Exception:
            continue

        # 收集梯度
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            k = name.lower()
            layer_idx = _extract_layer_idx(k)
            if layer_idx is None or layer_idx >= n_layers:
                continue

            grad_abs = param.grad.float().abs()

            if "q_proj" in k and "weight" in k:
                for h in range(n_heads):
                    start = h * head_dim
                    end = start + head_dim
                    if end <= grad_abs.shape[0]:
                        head_scores[layer_idx][h] += grad_abs[start:end].mean().item() / n

            elif ("gate_proj" in k or "up_proj" in k) and "weight" in k:
                for ni in range(min(intermediate, grad_abs.shape[0])):
                    neuron_scores[layer_idx][ni] += grad_abs[ni].mean().item() / n

            layer_scores[layer_idx] += grad_abs.mean().item() / n

    model.eval()
    return head_scores, neuron_scores, layer_scores


def _activation_scores(model, tokenizer, texts, n_layers, n_heads, head_dim, intermediate,
                       task=None, progress_base=62, progress_range=20):
    """基于激活值统计的重要性: 衡量实际使用频率"""
    import torch

    head_scores = [[0.0] * n_heads for _ in range(n_layers)]
    neuron_scores = [[0.0] * intermediate for _ in range(n_layers)]
    layer_scores = [0.0] * n_layers

    # 注册 hook 收集激活
    hooks = []
    activation_data = {}

    def _make_hook(layer_idx, module_type):
        def hook_fn(module, input, output):
            key = f"{layer_idx}_{module_type}"
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            if out is not None:
                activation_data[key] = out.detach().float().abs().mean(dim=(0, 1))
        return hook_fn

    # 找到各层并注册 hook
    layers, _ = _find_model_layers(model)
    for li, layer in enumerate(layers):
        if li >= n_layers:
            break
        # 尝试 hook self_attn 和 mlp
        for name, mod in layer.named_modules():
            if "self_attn" == name or "attention" == name:
                h = mod.register_forward_hook(_make_hook(li, "attn"))
                hooks.append(h)
            elif "mlp" == name:
                h = mod.register_forward_hook(_make_hook(li, "mlp"))
                hooks.append(h)

    model.eval()
    n = min(len(texts), 64)  # 激活分析不需太多样本

    with torch.no_grad():
        for ti in range(n):
            if ti % 10 == 0:
                _safe_update(task, progress_base + progress_range * ti / n,
                             f"激活分析 {ti}/{n}")
            tokens = tokenizer(texts[ti], return_tensors="pt",
                               truncation=True, max_length=256, padding=False)
            try:
                model(**tokens)
            except Exception:
                continue

            # 汇总激活数据
            for key, act in activation_data.items():
                li_str, mtype = key.split("_", 1)
                li = int(li_str)
                if mtype == "attn" and act.numel() >= n_heads * head_dim:
                    for h in range(n_heads):
                        start = h * head_dim
                        end = min(start + head_dim, act.numel())
                        head_scores[li][h] += act[start:end].mean().item() / n
                elif mtype == "mlp":
                    for ni in range(min(intermediate, act.numel())):
                        neuron_scores[li][ni] += act[ni].item() / n
                layer_scores[li] += act.mean().item() / n
            activation_data.clear()

    # 清理 hook
    for h in hooks:
        h.remove()

    return head_scores, neuron_scores, layer_scores


# ════════════════════════════════════════════════════════════
#  2. 结构化剪枝 (Structured Pruning)
# ════════════════════════════════════════════════════════════

def structured_prune(
    model_path: str,
    output_name: str,
    importance: Optional[Dict] = None,
    calibration_data: str = "",
    head_prune_ratio: float = 0.25,
    neuron_prune_ratio: float = 0.20,
    layer_prune_count: int = 0,
    task=None,
) -> str:
    """结构化剪枝 — 砍掉摸鱼的 head、neuron、整层

    不同于非结构化剪枝（只设零），这是真的删除结构 → 模型变小。

    Args:
        importance:        重要性评分（None=自动计算）
        head_prune_ratio:  砍掉多少比例的 attention head (0~0.5)
        neuron_prune_ratio: 砍掉多少比例的 MLP neuron (0~0.5)
        layer_prune_count: 砍掉几层（0=不砍层）
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    _safe_update(task, 2, "加载模型...")
    # 预导入模型类（防止 LazyAutoMapping 失效）
    try:
        from core.safe_loader import ensure_model_importable
        ensure_model_importable(model_path)
    except Exception:
        pass
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32,
        trust_remote_code=True, device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    cfg = model.config

    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_heads)
    hidden = cfg.hidden_size
    head_dim = hidden // n_heads
    intermediate = getattr(cfg, "intermediate_size", hidden * 4)

    # 获取重要性分数
    if importance is None:
        _safe_update(task, 5, "计算重要性分数（需要校准数据）...")
        importance = score_importance(
            model_path, calibration_data or "",
            method="combined" if calibration_data else "magnitude",
            task=task,
        )

    head_scores = importance["head_scores"]
    neuron_scores = importance["neuron_scores"]
    layer_scores = importance["layer_scores"]

    # ── 1. 剪层 ──
    layers_to_remove = []
    if layer_prune_count > 0:
        # 保护首尾层（embedding/lm_head 连接），只砍中间
        removable = list(range(1, n_layers - 1))
        removable.sort(key=lambda i: layer_scores[i])
        layers_to_remove = removable[:min(layer_prune_count, len(removable))]
        log(f"✂️ 剪掉 {len(layers_to_remove)} 层: {layers_to_remove}")

    if layers_to_remove:
        _safe_update(task, 40, f"剪掉 {len(layers_to_remove)} 层...")
        _prune_layers(model, layers_to_remove)
        n_layers -= len(layers_to_remove)
        cfg.num_hidden_layers = n_layers
        model.config.num_hidden_layers = n_layers
        # 重新索引 scores
        keep_indices = [i for i in range(len(layer_scores)) if i not in layers_to_remove]
        head_scores = [head_scores[i] for i in keep_indices]
        neuron_scores = [neuron_scores[i] for i in keep_indices]
        layer_scores = [layer_scores[i] for i in keep_indices]

    # ── 2. 剪 Attention Head (设零 + 收缩) ──
    n_heads_to_prune = max(0, int(n_heads * head_prune_ratio))
    if n_heads_to_prune > 0:
        _safe_update(task, 55, f"每层剪 {n_heads_to_prune}/{n_heads} attention heads...")
        _prune_attention_heads(model, head_scores, n_heads_to_prune, head_dim)

    # ── 3. 剪 MLP Neuron (设零) ──
    n_neurons_to_prune = max(0, int(intermediate * neuron_prune_ratio))
    if n_neurons_to_prune > 0:
        _safe_update(task, 70, f"每层剪 {n_neurons_to_prune}/{intermediate} MLP neurons...")
        _prune_mlp_neurons(model, neuron_scores, n_neurons_to_prune)

    # 保存
    _safe_update(task, 85, "保存剪枝模型...")
    out_dir = Path(LORAS_DIR) / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    cfg.save_pretrained(out_dir)

    new_params = sum(p.numel() for p in model.parameters())
    # 保存剪枝元信息
    meta = {
        "type": "pruned", "source": model_path,
        "layers_removed": layers_to_remove,
        "head_prune_ratio": head_prune_ratio,
        "neuron_prune_ratio": neuron_prune_ratio,
        "params_after": new_params,
    }
    try:
        (out_dir / "forgex_prune_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    except Exception:
        pass

    del model
    gc.collect()
    _safe_update(task, 100, f"✅ 剪枝完成: {new_params/1e6:.0f}M | {out_dir.name}")
    return str(out_dir)


def _prune_layers(model, indices_to_remove):
    """物理删除 Transformer 层"""
    import torch.nn as nn
    layers, attr = _find_model_layers(model)
    keep = [l for i, l in enumerate(layers) if i not in indices_to_remove]
    parts = attr.split(".")
    obj = model
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], nn.ModuleList(keep))


def _prune_attention_heads(model, head_scores, n_prune, head_dim):
    """将低重要性 head 的权重设零（结构化稀疏）"""
    import torch
    layers, _ = _find_model_layers(model)
    for li, layer in enumerate(layers):
        if li >= len(head_scores):
            break
        scores = head_scores[li]
        # 找最弱的 head
        sorted_heads = sorted(range(len(scores)), key=lambda h: scores[h])
        to_zero = set(sorted_heads[:n_prune])

        for name, param in layer.named_parameters():
            k = name.lower()
            if ("q_proj" in k or "k_proj" in k or "v_proj" in k) and "weight" in k:
                with torch.no_grad():
                    for h in to_zero:
                        start = h * head_dim
                        end = min(start + head_dim, param.shape[0])
                        param[start:end] = 0.0
            elif "o_proj" in k and "weight" in k:
                with torch.no_grad():
                    for h in to_zero:
                        start = h * head_dim
                        end = min(start + head_dim, param.shape[1])
                        param[:, start:end] = 0.0


def _prune_mlp_neurons(model, neuron_scores, n_prune):
    """将低重要性 MLP neuron 设零"""
    import torch
    layers, _ = _find_model_layers(model)
    for li, layer in enumerate(layers):
        if li >= len(neuron_scores):
            break
        scores = neuron_scores[li]
        sorted_neurons = sorted(range(len(scores)), key=lambda n: scores[n])
        to_zero = set(sorted_neurons[:n_prune])

        for name, param in layer.named_parameters():
            k = name.lower()
            if ("gate_proj" in k or "up_proj" in k) and "weight" in k:
                with torch.no_grad():
                    for ni in to_zero:
                        if ni < param.shape[0]:
                            param[ni] = 0.0
            elif "down_proj" in k and "weight" in k:
                with torch.no_grad():
                    for ni in to_zero:
                        if ni < param.shape[1]:
                            param[:, ni] = 0.0


# ════════════════════════════════════════════════════════════
#  3. 自蒸馏 (Self-Distillation)
# ════════════════════════════════════════════════════════════

def self_distill(
    teacher_path: str,
    student_path: str,
    calibration_data: str,
    output_name: str,
    temperature: float = 3.0,
    alpha_ce: float = 0.5,
    alpha_kd: float = 0.5,
    n_samples: int = 2000,
    lr: float = 5e-5,
    epochs: float = 2.0,
    batch_size: int = 2,
    max_seq_len: int = 1024,
    task=None,
) -> str:
    """自蒸馏: 训练好的大模型教自己的精简版

    关键: 用软标签（soft labels）传递"暗知识"（dark knowledge），
    比硬标签（one-hot）信息量大 10-100 倍。

    流程:
      1. Teacher 对校准数据生成 logits 分布
      2. Student 同时学:
         - 硬标签 (CE loss): 学正确答案
         - 软标签 (KD loss): 学 teacher 的概率分布（包含类间关系）

    Args:
        teacher_path: 训练好的大模型（或膨胀+训练后的模型）
        student_path: 精简后的小模型（或原始未膨胀模型）
        temperature:  蒸馏温度（越高→概率分布越软→传递越多暗知识）
        alpha_ce:     硬标签权重
        alpha_kd:     软标签权重
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.utils.data import DataLoader, Dataset

    _safe_update(task, 2, "加载 Teacher 模型...")
    # 预导入模型类（防止 LazyAutoMapping 失效）
    try:
        from core.safe_loader import ensure_model_importable
        ensure_model_importable(teacher_path)
    except Exception:
        pass
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_path, torch_dtype=torch.float16,
        trust_remote_code=True, device_map="auto",
    )
    teacher.eval()

    _safe_update(task, 8, "加载 Student 模型...")
    student = AutoModelForCausalLM.from_pretrained(
        student_path, torch_dtype=torch.float32,
        trust_remote_code=True, device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 如果有 CUDA，student 也放 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        student = student.to(device)

    _safe_update(task, 15, "准备蒸馏数据...")
    texts = _load_calibration_texts(calibration_data, n_samples)
    if not texts:
        raise ValueError("没有校准数据用于自蒸馏")

    # Tokenize
    all_ids = []
    for text in texts:
        enc = tokenizer(text, truncation=True, max_length=max_seq_len,
                        return_tensors="pt", padding=False)
        if enc["input_ids"].shape[1] >= 4:
            all_ids.append(enc["input_ids"].squeeze(0))

    log(f"蒸馏数据: {len(all_ids)} 条")

    # 训练
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    student.train()
    T = temperature
    total_steps = int(len(all_ids) * epochs / max(batch_size, 1))
    step = 0

    _safe_update(task, 20, f"开始自蒸馏: {total_steps} steps...")

    for ep in range(max(1, int(epochs))):
        import random
        random.shuffle(all_ids)

        for bi in range(0, len(all_ids), batch_size):
            batch = all_ids[bi:bi + batch_size]
            if not batch:
                continue

            # Pad batch
            max_len = max(ids.shape[0] for ids in batch)
            pad_id = tokenizer.pad_token_id or 0
            input_ids = torch.stack([
                F.pad(ids, (0, max_len - ids.shape[0]), value=pad_id)
                for ids in batch
            ]).to(device)

            labels = input_ids.clone()
            labels[labels == pad_id] = -100

            # Teacher forward (no grad)
            with torch.no_grad():
                t_out = teacher(input_ids=input_ids.to(teacher.device))
                t_logits = t_out.logits.to(device).float()

            # Student forward
            s_out = student(input_ids=input_ids, labels=labels)
            s_logits = s_out.logits
            ce_loss = s_out.loss

            # KD loss: KL divergence on soft distributions
            # 只在有效 token 上计算
            vocab_min = min(s_logits.shape[-1], t_logits.shape[-1])
            s_soft = F.log_softmax(s_logits[..., :vocab_min] / T, dim=-1)
            t_soft = F.softmax(t_logits[..., :vocab_min] / T, dim=-1)
            kd_loss = F.kl_div(s_soft, t_soft, reduction="batchmean") * (T * T)

            loss = alpha_ce * ce_loss + alpha_kd * kd_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            if step % 20 == 0:
                _safe_update(task, 20 + 65 * step / max(total_steps, 1),
                             f"Step {step}/{total_steps} | CE={ce_loss:.3f} KD={kd_loss:.3f}")

    # 保存
    _safe_update(task, 88, "保存蒸馏后模型...")
    out_dir = Path(LORAS_DIR) / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    meta = {
        "type": "self_distilled",
        "teacher": teacher_path, "student_base": student_path,
        "temperature": T, "alpha_ce": alpha_ce, "alpha_kd": alpha_kd,
        "steps": step, "samples": len(all_ids),
    }
    try:
        (out_dir / "forgex_distill_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    except Exception:
        pass

    del teacher, student
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _safe_update(task, 100, f"✅ 自蒸馏完成: {out_dir.name}")
    return str(out_dir)


# ════════════════════════════════════════════════════════════
#  4. 困难样本挖掘 (Hard Example Mining)
# ════════════════════════════════════════════════════════════

def mine_hard_examples(
    model_path: str,
    dataset_path: str,
    output_name: str,
    top_ratio: float = 0.3,
    n_samples: int = 0,
    task=None,
) -> str:
    """找出模型最薄弱的样本，用于集中再训练

    对每条数据:
      1. 用模型计算 loss
      2. loss 越高 = 模型越不会 = 越值得训练
      3. 取 top_ratio 最难的样本输出

    也生成 difficulty_distribution 报告。
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _safe_update(task, 2, "加载模型...")
    # 预导入模型类（防止 LazyAutoMapping 失效）
    try:
        from core.safe_loader import ensure_model_importable
        ensure_model_importable(model_path)
    except Exception:
        pass
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16,
        trust_remote_code=True, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    _safe_update(task, 10, "加载数据集...")
    data = _load_sft_data(dataset_path)
    if n_samples > 0:
        data = data[:n_samples]
    if not data:
        raise ValueError("数据集为空")

    # 逐条计算 loss
    _safe_update(task, 15, f"分析 {len(data)} 条数据的难度...")
    scored = []

    for i, item in enumerate(data):
        if i % 50 == 0:
            _safe_update(task, 15 + 55 * i / len(data),
                         f"分析 {i}/{len(data)}")

        text = _format_sft_text(item, tokenizer)
        tokens = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=1024, padding=False)
        input_ids = tokens["input_ids"].to(model.device)

        if input_ids.shape[1] < 4:
            continue

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss_val = outputs.loss.item() if outputs.loss is not None else 0.0

        scored.append({**item, "_loss": loss_val, "_idx": i})

    # 排序: loss 高 → 难
    scored.sort(key=lambda x: x["_loss"], reverse=True)

    # 取最难的
    n_hard = max(1, int(len(scored) * top_ratio))
    hard_examples = scored[:n_hard]
    easy_examples = scored[n_hard:]

    # 统计
    losses = [s["_loss"] for s in scored]
    stats = {
        "total": len(scored),
        "hard_count": len(hard_examples),
        "easy_count": len(easy_examples),
        "loss_mean": sum(losses) / len(losses) if losses else 0,
        "loss_p25": losses[int(len(losses) * 0.75)] if losses else 0,
        "loss_p50": losses[int(len(losses) * 0.50)] if losses else 0,
        "loss_p75": losses[int(len(losses) * 0.25)] if losses else 0,
        "loss_max": losses[0] if losses else 0,
        "loss_min": losses[-1] if losses else 0,
    }

    # 保存
    _safe_update(task, 78, "保存困难样本...")
    out_dir = Path(DATASETS_DIR) / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 困难样本（用于再训练）
    hard_clean = [{k: v for k, v in item.items() if not k.startswith("_")} for item in hard_examples]
    (out_dir / "hard_examples.json").write_text(
        json.dumps(hard_clean, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # 全量带分数（用于分析）
    (out_dir / "scored_all.json").write_text(
        json.dumps(scored, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # 统计报告
    (out_dir / "difficulty_report.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    del model
    gc.collect()

    _safe_update(task, 100,
        f"✅ 挖掘完成: {len(hard_examples)} 困难 / {len(easy_examples)} 简单 | {out_dir.name}")
    return str(out_dir / "hard_examples.json")


# ════════════════════════════════════════════════════════════
#  5. 锻压循环 (Forge-Press Cycle)
# ════════════════════════════════════════════════════════════

def forge_press_cycle(
    student_model: str,
    api_config_dict: Dict,
    calibration_data: str,
    output_name: str = "forged_1b",
    n_cycles: int = 3,
    # 膨胀参数
    expand_method: str = "depth",
    expand_extra_layers: int = 4,
    expand_target_hidden: int = 0,
    # 剪枝参数
    prune_heads: float = 0.2,
    prune_neurons: float = 0.15,
    prune_layers: int = 0,
    # 自蒸馏参数
    distill_temperature: float = 3.0,
    distill_epochs: float = 2.0,
    # 困难挖掘
    hard_ratio: float = 0.3,
    task=None,
) -> str:
    """完整锻压循环: 反复 expand→train→prune→distill→mine→retrain

    每一轮:
      Phase A: 膨胀 — 给模型长身体
      Phase B: 灌注 — API 教师灌知识（调用 api_teacher.train_student）
      Phase C: 评分 — 量化每个参数的贡献
      Phase D: 锻压 — 剪掉摸鱼的结构
      Phase E: 自蒸馏 — 大版本教小版本（软标签浓缩暗知识）
      Phase F: 困难挖掘 — 找出薄弱点
      Phase G: 精修 — 用困难样本再训练

    每轮循环后，剪枝力度逐渐加大:
      Round 1: 温和剪枝 (15% head, 10% neuron)
      Round 2: 中等剪枝 (20% head, 15% neuron)
      Round 3: 激进剪枝 (25% head, 20% neuron) + 剪层

    Returns: 最终模型路径
    """
    _safe_update(task, 0, f"🔥 锻压循环启动: {n_cycles} 轮")

    current_model = student_model
    original_model = student_model  # 保留原始模型用于自蒸馏的 student

    per_cycle = 100.0 / max(n_cycles, 1)

    for cycle in range(n_cycles):
        cycle_base = cycle * per_cycle
        cycle_name = f"cycle{cycle+1}"
        _safe_update(task, cycle_base,
                     f"═══ 锻压第 {cycle+1}/{n_cycles} 轮 ═══")

        # 逐轮加大剪枝力度
        cycle_head_prune = min(0.5, prune_heads + cycle * 0.05)
        cycle_neuron_prune = min(0.5, prune_neurons + cycle * 0.05)
        cycle_layer_prune = prune_layers if cycle >= n_cycles - 1 else 0

        # ── Phase A: 膨胀 ──
        expanded_name = f"{output_name}_{cycle_name}_expanded"
        _safe_update(task, cycle_base + per_cycle * 0.02,
                     f"[{cycle+1}A] 膨胀...")
        try:
            from core.expansion import depth_expand, width_expand, hybrid_expand

            if expand_method == "depth" and expand_extra_layers > 0:
                current_model = depth_expand(
                    current_model, expanded_name,
                    num_new_layers=expand_extra_layers,
                    noise_scale=0.01,
                )
            elif expand_method == "width" and expand_target_hidden > 0:
                current_model = width_expand(
                    current_model, expanded_name,
                    target_hidden=expand_target_hidden,
                    noise_scale=0.01,
                )
            elif expand_method in ("hybrid", "depth+width"):
                from transformers import AutoConfig
                try:
                    cfg = AutoConfig.from_pretrained(current_model, trust_remote_code=True)
                    tgt_layers = cfg.num_hidden_layers + expand_extra_layers
                except Exception:
                    tgt_layers = 24
                current_model = hybrid_expand(
                    current_model, expanded_name,
                    target_layers=tgt_layers,
                    target_hidden=expand_target_hidden,
                    noise_scale=0.01,
                )
            log(f"[{cycle+1}A] 膨胀完成: {current_model}")
        except Exception as e:
            log(f"[{cycle+1}A] 膨胀跳过: {e}")

        # ── Phase B: 灌注 (API 蒸馏) ──
        trained_name = f"{output_name}_{cycle_name}_trained"
        _safe_update(task, cycle_base + per_cycle * 0.15,
                     f"[{cycle+1}B] API 灌注...")
        try:
            from core.distiller import APITeacherConfig, api_teacher

            train_cfg = APITeacherConfig(**api_config_dict)
            train_cfg.student_model = current_model
            train_cfg.output_name = trained_name
            train_cfg.expand_before_train = False  # 已经膨胀过了

            result = api_teacher.train_student(train_cfg)
            current_model = str(Path(LORAS_DIR) / trained_name)
            if Path(result).is_dir():
                current_model = str(result)
            log(f"[{cycle+1}B] 灌注完成: {current_model}")
        except Exception as e:
            log(f"[{cycle+1}B] 灌注失败: {e}")

        # ── Phase C+D: 评分 + 剪枝 ──
        pruned_name = f"{output_name}_{cycle_name}_pruned"
        _safe_update(task, cycle_base + per_cycle * 0.50,
                     f"[{cycle+1}C] 评分+剪枝 (head {cycle_head_prune:.0%}, neuron {cycle_neuron_prune:.0%})...")
        try:
            current_model = structured_prune(
                current_model, pruned_name,
                calibration_data=calibration_data,
                head_prune_ratio=cycle_head_prune,
                neuron_prune_ratio=cycle_neuron_prune,
                layer_prune_count=cycle_layer_prune,
            )
            log(f"[{cycle+1}D] 剪枝完成: {current_model}")
        except Exception as e:
            log(f"[{cycle+1}D] 剪枝跳过: {e}")

        # ── Phase E: 自蒸馏 ──
        distilled_name = f"{output_name}_{cycle_name}_distilled"
        _safe_update(task, cycle_base + per_cycle * 0.65,
                     f"[{cycle+1}E] 自蒸馏 (T={distill_temperature})...")
        try:
            # Teacher = 当前（剪枝后但仍较大的）模型
            # Student = 原始模型（或上轮的输出）
            self_distill_student = original_model if cycle == 0 else current_model
            current_model = self_distill(
                teacher_path=current_model,
                student_path=self_distill_student,
                calibration_data=calibration_data,
                output_name=distilled_name,
                temperature=distill_temperature,
                epochs=distill_epochs,
            )
            log(f"[{cycle+1}E] 自蒸馏完成: {current_model}")
        except Exception as e:
            log(f"[{cycle+1}E] 自蒸馏跳过: {e}")

        # ── Phase F+G: 困难挖掘 + 精修 ──
        _safe_update(task, cycle_base + per_cycle * 0.82,
                     f"[{cycle+1}F] 困难样本挖掘...")
        try:
            hard_data_path = mine_hard_examples(
                current_model, calibration_data,
                output_name=f"{output_name}_{cycle_name}_hard",
                top_ratio=hard_ratio,
            )
            log(f"[{cycle+1}F] 困难样本: {hard_data_path}")

            # 用困难样本做一轮精修 SFT
            _safe_update(task, cycle_base + per_cycle * 0.90,
                         f"[{cycle+1}G] 困难精修...")
            from core.trainer import trainer_engine
            retrained_name = f"{output_name}_{cycle_name}_refined"
            refined = trainer_engine.train(
                method="sft", backend="auto",
                base_model=current_model,
                dataset_path=hard_data_path,
                params={
                    "output_name": retrained_name,
                    "lr": 5e-5,  # 低学习率精修
                    "batch_size": 1,
                    "epochs": 1,
                    "max_seq_len": 1024,
                    "rank": 32,
                },
            )
            current_model = str(Path(LORAS_DIR) / retrained_name)
            log(f"[{cycle+1}G] 精修完成: {current_model}")
        except Exception as e:
            log(f"[{cycle+1}F/G] 困难精修跳过: {e}")

    # 最终输出
    _safe_update(task, 98, "保存最终模型...")
    final_dir = Path(LORAS_DIR) / output_name
    if str(final_dir) != current_model:
        import shutil
        if final_dir.exists():
            shutil.rmtree(final_dir, ignore_errors=True)
        try:
            shutil.copytree(current_model, str(final_dir))
        except Exception:
            final_dir = Path(current_model)

    _safe_update(task, 100, f"🔥 锻压循环完成: {n_cycles} 轮 | {final_dir.name}")
    return str(final_dir)


# ════════════════════════════════════════════════════════════
#  工具函数
# ════════════════════════════════════════════════════════════

def _find_model_layers(model):
    """找到模型的 Transformer 层列表"""
    candidates = [
        ("model.layers", lambda m: m.model.layers),
        ("model.model.layers", lambda m: m.model.model.layers),
        ("transformer.h", lambda m: m.transformer.h),
        ("gpt_neox.layers", lambda m: m.gpt_neox.layers),
    ]
    for attr_path, getter in candidates:
        try:
            layers = getter(model)
            if hasattr(layers, "__len__") and len(layers) > 0:
                return layers, attr_path
        except (AttributeError, TypeError):
            continue
    raise RuntimeError("无法找到模型层")


def _extract_layer_idx(key: str) -> Optional[int]:
    """从参数名提取层号"""
    import re
    m = re.search(r'\.layers?\.(\d+)\.', key)
    if m:
        return int(m.group(1))
    m = re.search(r'\.h\.(\d+)\.', key)
    if m:
        return int(m.group(1))
    return None


def _norm(val, all_scores):
    """归一化到 0~1"""
    flat = []
    if isinstance(all_scores, list):
        for item in all_scores:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
    if not flat:
        return 0.0
    mn, mx = min(flat), max(flat)
    if mx - mn < 1e-10:
        return 0.5
    return (val - mn) / (mx - mn)


def _load_calibration_texts(path: str, n: int = 128) -> List[str]:
    """加载校准文本"""
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        # 尝试 datasets 目录
        p = Path(DATASETS_DIR) / path
    if not p.exists():
        return []

    texts = []
    try:
        if p.suffix in (".json", ".jsonl"):
            if p.suffix == ".jsonl":
                for line in p.read_text(encoding="utf-8").strip().split("\n"):
                    if line.strip():
                        item = json.loads(line)
                        texts.append(_item_to_text(item))
            else:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for item in data:
                        texts.append(_item_to_text(item))
        elif p.suffix == ".txt":
            texts = [t.strip() for t in p.read_text(encoding="utf-8").split("\n") if t.strip()]
    except Exception as e:
        log(f"加载校准数据失败: {e}")

    return texts[:n]


def _item_to_text(item: Dict) -> str:
    """把 SFT 格式转为纯文本"""
    if isinstance(item, str):
        return item
    parts = []
    if item.get("instruction"):
        parts.append(item["instruction"])
    if item.get("input"):
        parts.append(item["input"])
    if item.get("output"):
        parts.append(item["output"])
    if item.get("text"):
        parts.append(item["text"])
    return "\n".join(parts) if parts else str(item)


def _load_sft_data(path: str) -> List[Dict]:
    """加载 SFT 数据"""
    p = Path(path)
    if not p.exists():
        p = Path(DATASETS_DIR) / path
    if not p.exists():
        return []
    try:
        if p.suffix == ".jsonl":
            data = []
            for line in p.read_text(encoding="utf-8").strip().split("\n"):
                if line.strip():
                    data.append(json.loads(line))
            return data
        else:
            raw = json.loads(p.read_text(encoding="utf-8"))
            return raw if isinstance(raw, list) else [raw]
    except Exception:
        return []


def _format_sft_text(item: Dict, tokenizer) -> str:
    """格式化 SFT 数据为训练文本"""
    if "text" in item:
        return item["text"]
    instruction = item.get("instruction", "")
    inp = item.get("input", "")
    output = item.get("output", "")
    if hasattr(tokenizer, "apply_chat_template"):
        msgs = [{"role": "user", "content": f"{instruction}\n{inp}".strip()},
                {"role": "assistant", "content": output}]
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False)
        except Exception:
            pass
    return f"### Instruction:\n{instruction}\n### Input:\n{inp}\n### Response:\n{output}"
