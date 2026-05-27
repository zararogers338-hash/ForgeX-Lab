# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

# core/molora.py - Mixture of LoRA Experts (MoLoRA)
#
# 核心: 一个模型内部 N 组 LoRA 专家 + 小门控网络
#   训练时: 门控自动学习路由
#   推理时: 门控 → Top-K → 稀疏计算
#   导出时: 合并进基座 → 标准模型, 兼容所有推理软件
#
# 每个目标 Linear 层:
#   Input → Gate(MLP) → Top-K
#   Expert_1: x @ A1^T @ B1^T  ──┐
#   Expert_2: x @ A2^T @ B2^T  ──┼→ weighted sum → output
#   Expert_3: x @ A3^T @ B3^T  ──┘

import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from core.logger import log


class MoLoRALinear(nn.Module):
    """
    替换 nn.Linear, 加入 N 组 LoRA 专家 + 门控路由.
    
    Standard LoRA:  y = Wx + BAx * s
    MoLoRA:         y = Wx + Σ(g_i * B_i A_i x) * s
                    where g = TopK(softmax(Gate(pool(x))))
    """

    def __init__(self, base_linear, n_experts=4, rank=16,
                 alpha=32.0, top_k=2, dropout=0.05):
        super().__init__()
        self.base_linear = base_linear
        self.n_experts = n_experts
        self.rank = rank
        self.top_k = min(top_k, n_experts)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_f = base_linear.in_features
        out_f = base_linear.out_features

        # 冻结基座
        base_linear.weight.requires_grad = False
        if base_linear.bias is not None:
            base_linear.bias.requires_grad = False

        # N 组 LoRA (A: kaiming, B: zero → 初始 LoRA=0)
        self.lora_A = nn.ParameterList([
            nn.Parameter(torch.empty(rank, in_f)) for _ in range(n_experts)
        ])
        self.lora_B = nn.ParameterList([
            nn.Parameter(torch.zeros(out_f, rank)) for _ in range(n_experts)
        ])
        for a in self.lora_A:
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))

        # 门控 (极小: in_f → 32 → n_experts)
        gh = min(32, max(8, in_f // 128))
        self.gate = nn.Sequential(
            nn.Linear(in_f, gh, bias=False), nn.SiLU(),
            nn.Linear(gh, n_experts, bias=False),
        )

        self.register_buffer("expert_counts", torch.zeros(n_experts))
        self._aux_loss = torch.tensor(0.0)

    def forward(self, x):
        base_out = self.base_linear(x)
        dtype = x.dtype

        # 门控 (序列维度 mean pool)
        gate_in = x.float().mean(dim=-2) if x.dim() >= 3 else x.float()
        gate_logits = self.gate(gate_in)
        topk_logits, topk_idx = gate_logits.topk(self.top_k, dim=-1)
        topk_w = F.softmax(topk_logits, dim=-1)

        # 负载均衡损失
        if self.training:
            probs = F.softmax(gate_logits, dim=-1)
            mask = F.one_hot(topk_idx, self.n_experts).float().sum(1)
            self._aux_loss = (mask.mean(0) * probs.mean(0)).sum() * self.n_experts

        # 稀疏专家计算
        xd = self.dropout(x)
        lora_out = torch.zeros_like(base_out)

        for k in range(self.top_k):
            eidx = topk_idx[:, k]
            ew = topk_w[:, k]
            for e in range(self.n_experts):
                m = (eidx == e)
                if not m.any():
                    continue
                xs = xd[m]
                h = F.linear(F.linear(xs, self.lora_A[e]), self.lora_B[e])
                w = ew[m]
                if h.dim() == 3:
                    h = h * w.unsqueeze(-1).unsqueeze(-1)
                else:
                    h = h * w.unsqueeze(-1)
                lora_out[m] = lora_out[m] + h * self.scaling
                if self.training:
                    self.expert_counts[e] += m.sum().item()

        return base_out + lora_out.to(dtype)

    def get_aux_loss(self):
        return self._aux_loss

    def get_expert_usage(self):
        t = self.expert_counts.sum().item()
        if t == 0:
            return {i: 0.0 for i in range(self.n_experts)}
        return {i: self.expert_counts[i].item() / t for i in range(self.n_experts)}

    def merge_to_base(self, weights=None):
        """合并专家进基座 → 标准 nn.Linear"""
        if weights is None:
            t = self.expert_counts.sum().item()
            if t > 0:
                weights = [self.expert_counts[i].item() / t for i in range(self.n_experts)]
            else:
                weights = [1.0 / self.n_experts] * self.n_experts

        delta = torch.zeros_like(self.base_linear.weight.data)
        for i, w in enumerate(weights):
            delta += w * (self.lora_B[i].data @ self.lora_A[i].data) * self.scaling

        self.base_linear.weight.data += delta.to(self.base_linear.weight.dtype)
        return self.base_linear


# ═══════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════

@dataclass
class MoLoRAConfig:
    n_experts: int = 4
    rank: int = 16
    alpha: float = 32.0
    top_k: int = 2
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    aux_loss_weight: float = 0.01
    expert_labels: List[str] = field(default_factory=lambda: [
        "专家1", "专家2", "专家3", "专家4"])

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════
# 注入 / 合并 / 辅助函数
# ═══════════════════════════════════════════════════

def apply_molora(model, config):
    """给模型注入 MoLoRA 层 — 替换目标 Linear"""
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        short = name.split(".")[-1]
        if short not in config.target_modules and "all-linear" not in config.target_modules:
            continue
        # 跳过 lm_head / embed
        if "lm_head" in name or "embed" in name:
            continue

        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)

        ml = MoLoRALinear(module, config.n_experts, config.rank,
                          config.alpha, config.top_k, config.dropout)
        setattr(parent, parts[-1], ml)
        replaced += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"🧠 MoLoRA: {replaced} 层 | {config.n_experts} 专家 × r{config.rank} | "
        f"Top-{config.top_k} | trainable {trainable:,}/{total:,} ({trainable/total*100:.2f}%)")
    model._molora_config = config
    return model


def collect_aux_loss(model):
    """收集所有 MoLoRA 层的负载均衡损失"""
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    n = 0
    for m in model.modules():
        if isinstance(m, MoLoRALinear):
            total = total + m.get_aux_loss()
            n += 1
    return total / max(n, 1)


def get_expert_usage_summary(model, labels=None):
    """可读的专家使用摘要"""
    totals = {}
    n_layers = 0
    for _, m in model.named_modules():
        if isinstance(m, MoLoRALinear):
            for eid, frac in m.get_expert_usage().items():
                totals[eid] = totals.get(eid, 0) + frac
            n_layers += 1
    if n_layers == 0:
        return "无 MoLoRA 层"
    lines = ["📊 专家激活统计 (训练期间):"]
    for eid in sorted(totals):
        pct = totals[eid] / n_layers * 100
        label = labels[eid] if labels and eid < len(labels) else f"专家{eid}"
        bar = "█" * int(pct / 2.5) + "░" * (40 - int(pct / 2.5))
        lines.append(f"  {label}: {pct:5.1f}% |{bar}|")
    return "\n".join(lines)


def merge_molora_to_base(model, strategy="usage", custom_weights=None):
    """
    将 MoLoRA 合并进基座 → 标准模型, 兼容所有推理软件.
    
    strategy: "usage" (按训练统计) / "uniform" / "custom"
    """
    merged = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, MoLoRALinear):
            continue
        if strategy == "uniform":
            w = [1.0 / module.n_experts] * module.n_experts
        elif strategy == "custom" and custom_weights:
            w = [custom_weights.get(i, 1.0 / module.n_experts) for i in range(module.n_experts)]
        else:
            w = None

        linear = module.merge_to_base(w)
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], linear)
        merged += 1

    log(f"✅ MoLoRA → 标准模型: {merged} 层合并完成 (strategy={strategy})")
    return model


# ═══════════════════════════════════════════════════
# MoLoRA Trainer (标准 loss + 负载均衡)
# ═══════════════════════════════════════════════════

def make_molora_trainer_class(base_cls, aux_weight=0.01):
    """创建 MoLoRA Trainer — 在标准 loss 上加负载均衡损失"""
    class MoLoRATrainer(base_cls):
        def __init__(self, *args, molora_aux_w=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._aux_w = molora_aux_w or aux_weight

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            aux = collect_aux_loss(model)
            if aux.item() > 0:
                loss = loss + self._aux_w * aux
            return (loss, outputs) if return_outputs else loss

    MoLoRATrainer.__name__ = f"MoLoRA{base_cls.__name__}"
    return MoLoRATrainer


# ═══════════════════════════════════════════════════
# Checkpoint 保存/加载
# ═══════════════════════════════════════════════════

def save_molora_checkpoint(model, path, config):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state = {}
    for name, m in model.named_modules():
        if not isinstance(m, MoLoRALinear):
            continue
        for i in range(m.n_experts):
            state[f"{name}.lora_A.{i}"] = m.lora_A[i].data.cpu()
            state[f"{name}.lora_B.{i}"] = m.lora_B[i].data.cpu()
        for gn, gp in m.gate.named_parameters():
            state[f"{name}.gate.{gn}"] = gp.data.cpu()
        state[f"{name}.expert_counts"] = m.expert_counts.cpu()

    torch.save(state, path / "molora_weights.pt")
    (path / "molora_config.json").write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    (path / "expert_usage.txt").write_text(
        get_expert_usage_summary(model, config.expert_labels), encoding="utf-8")
    log(f"💾 MoLoRA 已保存: {path}")


def load_molora_checkpoint(model, path):
    path = Path(path)
    config = MoLoRAConfig.from_dict(
        json.loads((path / "molora_config.json").read_text(encoding="utf-8")))
    model = apply_molora(model, config)
    state = torch.load(path / "molora_weights.pt", map_location="cpu")
    for name, m in model.named_modules():
        if not isinstance(m, MoLoRALinear):
            continue
        for i in range(m.n_experts):
            ka, kb = f"{name}.lora_A.{i}", f"{name}.lora_B.{i}"
            if ka in state: m.lora_A[i].data.copy_(state[ka])
            if kb in state: m.lora_B[i].data.copy_(state[kb])
        for gn, gp in m.gate.named_parameters():
            kg = f"{name}.gate.{gn}"
            if kg in state: gp.data.copy_(state[kg])
        kc = f"{name}.expert_counts"
        if kc in state: m.expert_counts.copy_(state[kc])
    log(f"📂 MoLoRA 已加载: {path}")
    return model, config
