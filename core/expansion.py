# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

"""ForgeX 知识膨胀引擎 — 给模型"塞知识"增加参数

不考虑内存约束，全力扩展:
  1. 层复制 (Depth Expansion)  — 复制 Transformer 层使模型更深
  2. 宽度膨胀 (Width Expansion) — Net2Net 扩展 hidden_size / MLP / attention heads
  3. 词表扩展 (Vocab Expansion) — 添加新 token 并用语义相近 token 初始化
  4. 知识嫁接 (Knowledge Graft) — 从大模型"偷"知识层插入小模型
  5. 混合膨胀 (Hybrid)          — 同时扩深+扩宽，一步到位

所有操作保留原始知识（不是随机初始化），输出可直接继续训练。
"""

from __future__ import annotations

import copy
import gc
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core import LORAS_DIR, log


def _safe_update(task, p, msg):
    if task is not None:
        try:
            task.update_progress(float(p), str(msg))
        except Exception:
            pass


def _copy_custom_model_files(src_dir: str, dst_dir: str):
    """复制自定义模型文件（modeling_*.py, configuration_*.py 等）
    
    Phi-3, InternLM 等模型依赖 trust_remote_code=True，
    save_pretrained 不会复制这些 .py 文件，需要手动复制。
    """
    import shutil
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    if not src_path.is_dir():
        return
    for f in src_path.iterdir():
        if f.suffix == ".py" and not f.name.startswith("__"):
            dst_file = dst_path / f.name
            if not dst_file.exists():
                try:
                    shutil.copy2(f, dst_file)
                except Exception:
                    pass


# ════════════════════════════════════════════════════
#  层复制 (Depth Expansion)
# ════════════════════════════════════════════════════

def depth_expand(
    model_path: str,
    output_name: str,
    strategy: str = "repeat_middle",
    num_new_layers: int = 8,
    noise_scale: float = 0.01,
    task=None,
) -> str:
    """复制 Transformer 层使模型更深

    策略:
      - "repeat_middle": 复制中间层（最安全，中间层通常最通用）
      - "repeat_all":    均匀复制所有层（如 12→24，每层复制一次）
      - "repeat_top":    复制顶层（加强生成能力）
      - "repeat_bottom": 复制底层（加强特征提取）
      - "interleave":    交错插入（ABAB→AABBAB，渐进扩展）

    Args:
        model_path:     源模型路径
        output_name:    输出目录名
        strategy:       复制策略
        num_new_layers: 要添加多少层（实际可能根据策略调整）
        noise_scale:    给复制层加多少噪声避免完全相同（0.0-0.1）
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    _safe_update(task, 2, f"加载模型: {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # 预导入模型类（防止 LazyAutoMapping 失效）
    try:
        from core.safe_loader import ensure_model_importable
        ensure_model_importable(model_path)
    except Exception:
        pass
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32,  # fp32 精度操作权重
        trust_remote_code=True, device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 找到 layers 模块
    layers, layers_attr = _find_layers(model)
    orig_n = len(layers)
    log(f"原始: {orig_n} 层, {_count_params(model)/1e6:.0f}M 参数")
    _safe_update(task, 10, f"原始 {orig_n} 层 → 计划添加 {num_new_layers} 层")

    # 确定哪些层要复制
    source_indices = _select_layers_to_copy(orig_n, num_new_layers, strategy)
    log(f"复制策略 {strategy}: 源层索引 = {source_indices}")

    # 执行复制
    _safe_update(task, 20, f"复制 {len(source_indices)} 层...")
    new_layers_list = list(layers)
    insert_offset = 0

    for i, (src_idx, insert_pos) in enumerate(source_indices):
        _safe_update(task, 20 + 50 * i / len(source_indices),
                     f"复制层 {src_idx} → 位置 {insert_pos + insert_offset}")
        new_layer = copy.deepcopy(layers[src_idx])
        # 加微量噪声打破对称性
        if noise_scale > 0:
            with torch.no_grad():
                for p in new_layer.parameters():
                    p.add_(torch.randn_like(p) * noise_scale * p.std())
        new_layers_list.insert(insert_pos + insert_offset, new_layer)
        insert_offset += 1

    # 替换 layers
    import torch.nn as nn
    new_module = nn.ModuleList(new_layers_list)
    _set_layers(model, new_module, layers_attr)

    # 更新 config
    new_n = len(new_layers_list)
    config.num_hidden_layers = new_n
    model.config.num_hidden_layers = new_n

    _safe_update(task, 75, f"扩展完成: {orig_n} → {new_n} 层")
    log(f"扩展后: {new_n} 层, {_count_params(model)/1e6:.0f}M 参数")

    # 保存
    out_dir = Path(LORAS_DIR) / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    _safe_update(task, 80, "保存模型...")
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    config.save_pretrained(out_dir)
    _copy_custom_model_files(model_path, out_dir)

    _save_expansion_meta(out_dir, model_path, "depth_expand", {
        "strategy": strategy, "orig_layers": orig_n, "new_layers": new_n,
        "noise_scale": noise_scale, "source_indices": [s[0] for s in source_indices],
        "params_before": _count_params_from_dir(model_path),
        "params_after": _count_params(model),
    })

    del model
    gc.collect()
    _safe_update(task, 100, f"✅ 深度膨胀: {orig_n}→{new_n} 层 | {out_dir.name}")
    return str(out_dir)


def _select_layers_to_copy(n_layers, n_new, strategy):
    """返回 [(source_idx, insert_position), ...]"""
    results = []
    if strategy == "repeat_middle":
        mid = n_layers // 2
        start = mid - n_new // 2
        for i in range(n_new):
            src = start + (i % (n_new))
            src = max(0, min(src, n_layers - 1))
            results.append((src, mid + i))
    elif strategy == "repeat_all":
        step = max(1, n_layers // max(n_new, 1))
        for i in range(n_new):
            src = (i * step) % n_layers
            results.append((src, src + i + 1))
    elif strategy == "repeat_top":
        top_start = max(0, n_layers - n_new)
        for i in range(n_new):
            src = top_start + (i % (n_layers - top_start))
            results.append((src, n_layers + i))
    elif strategy == "repeat_bottom":
        for i in range(n_new):
            src = i % min(n_new, n_layers // 2)
            results.append((src, n_new + i))
    elif strategy == "interleave":
        step = max(1, n_layers // max(n_new, 1))
        for i in range(n_new):
            src = min((i + 1) * step, n_layers - 1)
            results.append((src, src + i + 1))
    else:
        # 默认: repeat_middle (直接内联，不递归)
        mid = n_layers // 2
        start = mid - n_new // 2
        for i in range(n_new):
            src = start + (i % (n_new))
            src = max(0, min(src, n_layers - 1))
            results.append((src, mid + i))
    return results


# ════════════════════════════════════════════════════
#  宽度膨胀 (Width Expansion) — Net2Net
# ════════════════════════════════════════════════════

def width_expand(
    model_path: str,
    output_name: str,
    target_hidden: int = 0,
    target_intermediate: int = 0,
    target_heads: int = 0,
    target_kv_heads: int = 0,
    noise_scale: float = 0.01,
    task=None,
) -> str:
    """Net2Net 宽度膨胀 — 扩展 hidden_size / MLP / attention heads

    核心思路: 把现有神经元"分裂"成两个（保留原始激活），
    而不是随机初始化新参数。

    Args:
        target_hidden:       目标隐藏维度（0=不变）
        target_intermediate: 目标 MLP 中间维度（0=自动 = hidden * 8/3 rounded）
        target_heads:        目标注意力头数（0=不变）
        target_kv_heads:     目标 KV 头数（0=不变）
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    _safe_update(task, 2, f"加载模型: {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32,
        trust_remote_code=True, device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    old_h = config.hidden_size
    old_inter = getattr(config, "intermediate_size", old_h * 4)
    old_heads = config.num_attention_heads
    old_kv = getattr(config, "num_key_value_heads", old_heads)

    new_h = target_hidden if target_hidden > old_h else old_h
    new_inter = target_intermediate if target_intermediate > old_inter else (
        int(new_h * 8 / 3 / 128) * 128 if target_hidden > old_h else old_inter
    )
    new_heads = target_heads if target_heads > old_heads else old_heads
    new_kv = target_kv_heads if target_kv_heads > old_kv else old_kv

    if new_h == old_h and new_inter == old_inter and new_heads == old_heads:
        raise ValueError("所有目标维度都不大于当前值，无需膨胀")

    log(f"宽度膨胀: hidden {old_h}→{new_h} | MLP {old_inter}→{new_inter} | "
        f"heads {old_heads}→{new_heads} | KV {old_kv}→{new_kv}")
    _safe_update(task, 10, f"hidden {old_h}→{new_h}, MLP {old_inter}→{new_inter}")

    sd = model.state_dict()
    new_sd = {}

    _safe_update(task, 15, "分析权重结构...")
    total_keys = len(sd)

    for ki, (key, tensor) in enumerate(sd.items()):
        if ki % 50 == 0:
            _safe_update(task, 15 + 60 * ki / total_keys, f"膨胀权重 {ki}/{total_keys}: {key}")

        new_tensor = _expand_tensor(
            key, tensor,
            old_h, new_h,
            old_inter, new_inter,
            old_heads, new_heads,
            old_kv, new_kv,
            noise_scale,
        )
        new_sd[key] = new_tensor

    _safe_update(task, 78, "更新配置并加载新权重...")
    # 更新 config
    config.hidden_size = new_h
    config.intermediate_size = new_inter
    config.num_attention_heads = new_heads
    config.num_key_value_heads = new_kv

    # 重建模型
    del model
    gc.collect()

    new_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    # Load expanded weights
    missing, unexpected = new_model.load_state_dict(new_sd, strict=False)
    if missing:
        log(f"⚠️ 缺少的 keys: {missing[:5]}...")
    if unexpected:
        log(f"⚠️ 多余的 keys: {unexpected[:5]}...")

    new_params = _count_params(new_model)
    _safe_update(task, 85, f"膨胀后: {new_params/1e6:.0f}M 参数")

    # 保存
    out_dir = Path(LORAS_DIR) / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    _safe_update(task, 88, "保存模型...")
    new_model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    config.save_pretrained(out_dir)
    _copy_custom_model_files(model_path, out_dir)

    _save_expansion_meta(out_dir, model_path, "width_expand", {
        "old_hidden": old_h, "new_hidden": new_h,
        "old_intermediate": old_inter, "new_intermediate": new_inter,
        "old_heads": old_heads, "new_heads": new_heads,
        "params_before": _count_params_from_dir(model_path),
        "params_after": new_params,
    })

    del new_model, new_sd
    gc.collect()
    _safe_update(task, 100, f"✅ 宽度膨胀完成 | {out_dir.name}")
    return str(out_dir)


def _expand_tensor(key, tensor, old_h, new_h, old_inter, new_inter,
                   old_heads, new_heads, old_kv, new_kv, noise):
    """Net2Net 式扩展单个张量

    规则: 根据 key 名判断这个张量属于哪个部分，然后做对应维度的膨胀。
    膨胀方式: 把原始神经元复制到新位置（保留知识），加微量噪声打破对称性。
    """
    import torch

    shape = list(tensor.shape)
    k = key.lower()

    # Embedding: [vocab, hidden] → [vocab, new_hidden]
    if "embed" in k and "weight" in k and len(shape) == 2:
        if shape[1] == old_h and new_h > old_h:
            return _pad_dim(tensor, 1, new_h, noise)
        return tensor

    # LM head: [vocab, hidden] → [vocab, new_hidden]
    if "lm_head" in k and len(shape) == 2:
        if shape[1] == old_h and new_h > old_h:
            return _pad_dim(tensor, 1, new_h, noise)
        return tensor

    # LayerNorm / RMSNorm: [hidden] → [new_hidden]
    if ("norm" in k or "layernorm" in k) and len(shape) == 1:
        if shape[0] == old_h and new_h > old_h:
            return _pad_dim(tensor, 0, new_h, noise)
        return tensor

    # Q projection: [num_heads * head_dim, hidden] → expand both dims
    if "q_proj" in k and "weight" in k:
        return _expand_qkv(tensor, old_h, new_h, old_heads, new_heads,
                           old_heads, new_heads, noise)
    if "k_proj" in k and "weight" in k:
        return _expand_qkv(tensor, old_h, new_h, old_kv, new_kv,
                           old_kv, new_kv, noise)
    if "v_proj" in k and "weight" in k:
        return _expand_qkv(tensor, old_h, new_h, old_kv, new_kv,
                           old_kv, new_kv, noise)
    if "o_proj" in k and "weight" in k:
        # o_proj: [hidden, num_heads * head_dim]
        t = tensor
        if shape[0] == old_h and new_h > old_h:
            t = _pad_dim(t, 0, new_h, noise)
        if shape[1] == old_heads * (old_h // old_heads):
            new_head_dim = new_h // new_heads
            target_1 = new_heads * new_head_dim
            if target_1 > shape[1]:
                t = _pad_dim(t, 1, target_1, noise)
        return t

    # Gate/Up/Down proj (MLP)
    if "gate_proj" in k or "up_proj" in k:
        # [intermediate, hidden]
        t = tensor
        if shape[1] == old_h and new_h > old_h:
            t = _pad_dim(t, 1, new_h, noise)
        if shape[0] == old_inter and new_inter > old_inter:
            t = _pad_dim(t, 0, new_inter, noise)
        return t
    if "down_proj" in k:
        # [hidden, intermediate]
        t = tensor
        if shape[0] == old_h and new_h > old_h:
            t = _pad_dim(t, 0, new_h, noise)
        if shape[1] == old_inter and new_inter > old_inter:
            t = _pad_dim(t, 1, new_inter, noise)
        return t

    # Bias vectors
    if "bias" in k and len(shape) == 1:
        if shape[0] == old_h and new_h > old_h:
            return _pad_dim(tensor, 0, new_h, noise)
        if shape[0] == old_inter and new_inter > old_inter:
            return _pad_dim(tensor, 0, new_inter, noise)

    return tensor


def _expand_qkv(tensor, old_h, new_h, old_n, new_n, old_n_total, new_n_total, noise):
    """扩展 QKV projection"""
    import torch
    shape = list(tensor.shape)
    t = tensor

    # dim 1 (input): hidden → new_hidden
    if shape[1] == old_h and new_h > old_h:
        t = _pad_dim(t, 1, new_h, noise)

    # dim 0 (output): n_heads * head_dim → new
    head_dim = old_h // old_n_total if old_n_total > 0 else old_h
    new_head_dim = new_h // new_n_total if new_n_total > 0 else new_h
    target_0 = new_n * new_head_dim
    if target_0 > shape[0]:
        t = _pad_dim(t, 0, target_0, noise)

    return t


def _pad_dim(tensor, dim, target_size, noise=0.01):
    """将张量某个维度从当前大小扩展到 target_size（复制 + 噪声）"""
    import torch

    current = tensor.shape[dim]
    if current >= target_size:
        return tensor

    extra = target_size - current
    # 从现有值中循环复制
    indices = torch.arange(extra) % current
    slices = [slice(None)] * tensor.ndim
    slices[dim] = indices
    pad_part = tensor[tuple(slices)].clone()

    # 加噪声
    if noise > 0:
        pad_part = pad_part + torch.randn_like(pad_part) * noise * pad_part.std().clamp(min=1e-6)

    return torch.cat([tensor, pad_part], dim=dim)


# ════════════════════════════════════════════════════
#  知识嫁接 (Knowledge Graft)
# ════════════════════════════════════════════════════

def knowledge_graft(
    small_model_path: str,
    large_model_path: str,
    output_name: str,
    graft_layers: str = "middle",  # "top", "middle", "bottom", "interleave"
    num_graft_layers: int = 4,
    noise_scale: float = 0.005,
    task=None,
) -> str:
    """从大模型"偷"层插入小模型

    前提: 两个模型 hidden_size 相同（如同系列不同大小的变体）。
    如果 hidden_size 不同，会先对大模型层做投影适配。

    典型用法:
      - Qwen2.5-0.5B (24层) + Qwen2.5-7B 的中间层 → 28层混合模型
      - 自训练的 100M 模型 + 公开 1B 模型的知识层
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig

    _safe_update(task, 2, f"加载小模型: {small_model_path}")
    small_config = AutoConfig.from_pretrained(small_model_path, trust_remote_code=True)
    small_model = AutoModelForCausalLM.from_pretrained(
        small_model_path, torch_dtype=torch.float32,
        trust_remote_code=True, device_map="cpu",
    )
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(small_model_path, trust_remote_code=True)

    _safe_update(task, 15, f"加载大模型: {large_model_path}")
    large_config = AutoConfig.from_pretrained(large_model_path, trust_remote_code=True)
    large_model = AutoModelForCausalLM.from_pretrained(
        large_model_path, torch_dtype=torch.float32,
        trust_remote_code=True, device_map="cpu",
    )

    small_layers, small_attr = _find_layers(small_model)
    large_layers, _ = _find_layers(large_model)

    s_h = small_config.hidden_size
    l_h = large_config.hidden_size

    log(f"小模型: {len(small_layers)}层 × {s_h}dim | 大模型: {len(large_layers)}层 × {l_h}dim")

    # 选择要嫁接的层
    n_large = len(large_layers)
    if graft_layers == "middle":
        start = n_large // 2 - num_graft_layers // 2
    elif graft_layers == "top":
        start = n_large - num_graft_layers
    elif graft_layers == "bottom":
        start = 0
    else:  # interleave
        start = n_large // 4

    graft_indices = list(range(max(0, start), min(n_large, start + num_graft_layers)))
    log(f"嫁接策略: {graft_layers}, 取大模型层 {graft_indices}")

    _safe_update(task, 30, f"提取 {len(graft_indices)} 层...")

    # 提取并适配层
    import torch.nn as nn
    grafted_layers = []
    for gi, idx in enumerate(graft_indices):
        _safe_update(task, 30 + 30 * gi / len(graft_indices),
                     f"适配层 {idx}...")
        layer = copy.deepcopy(large_layers[idx])

        if l_h != s_h:
            # 需要投影适配: 在层前后加线性投影
            layer = _ProjectedLayer(layer, l_h, s_h)

        # 加微量噪声
        if noise_scale > 0:
            with torch.no_grad():
                for p in layer.parameters():
                    if p.requires_grad:
                        p.add_(torch.randn_like(p) * noise_scale * p.std().clamp(min=1e-6))

        grafted_layers.append(layer)

    # 插入到小模型
    _safe_update(task, 65, "插入嫁接层...")
    new_layers_list = list(small_layers)
    insert_pos = len(small_layers) // 2  # 插在中间
    if graft_layers == "top":
        insert_pos = len(small_layers)
    elif graft_layers == "bottom":
        insert_pos = 0

    for i, gl in enumerate(grafted_layers):
        new_layers_list.insert(insert_pos + i, gl)

    new_module = nn.ModuleList(new_layers_list)
    _set_layers(small_model, new_module, small_attr)
    small_model.config.num_hidden_layers = len(new_layers_list)

    # 释放大模型
    del large_model, large_layers
    gc.collect()

    _safe_update(task, 80, "保存嫁接模型...")
    out_dir = Path(LORAS_DIR) / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    small_model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    small_model.config.save_pretrained(out_dir)
    _copy_custom_model_files(small_model_path, out_dir)
    _copy_custom_model_files(large_model_path, out_dir)  # 大模型可能也有自定义文件

    _save_expansion_meta(out_dir, small_model_path, "knowledge_graft", {
        "small_model": small_model_path, "large_model": large_model_path,
        "graft_layers": graft_layers, "graft_indices": graft_indices,
        "small_hidden": s_h, "large_hidden": l_h,
        "orig_layers": len(small_layers),
        "new_layers": len(new_layers_list),
        "params_after": _count_params(small_model),
    })

    del small_model
    gc.collect()
    _safe_update(task, 100,
        f"✅ 知识嫁接: {len(small_layers)}→{len(new_layers_list)} 层 | {out_dir.name}")
    return str(out_dir)


def _ProjectedLayer(large_layer, l_h, s_h):
    """当大小模型 hidden_size 不同时，用投影层适配（lazy torch import）"""
    import torch
    import torch.nn as nn

    class _Projected(nn.Module):
        def __init__(self, inner, large_dim, small_dim):
            super().__init__()
            self.proj_in = nn.Linear(small_dim, large_dim, bias=False)
            self.inner = inner
            self.proj_out = nn.Linear(large_dim, small_dim, bias=False)
            with torch.no_grad():
                min_dim = min(small_dim, large_dim)
                nn.init.eye_(self.proj_in.weight[:min_dim, :min_dim])
                nn.init.eye_(self.proj_out.weight[:min_dim, :min_dim])

        def forward(self, hidden_states, **kwargs):
            h = self.proj_in(hidden_states)
            out = self.inner(h, **kwargs)
            if isinstance(out, tuple):
                return (self.proj_out(out[0]),) + out[1:]
            return self.proj_out(out)

    return _Projected(large_layer, l_h, s_h)


# ════════════════════════════════════════════════════
#  词表扩展 (Vocab Expansion)
# ════════════════════════════════════════════════════

def vocab_expand(
    model_path: str,
    output_name: str,
    new_tokens: Optional[List[str]] = None,
    new_vocab_size: int = 0,
    init_strategy: str = "semantic",
    task=None,
) -> str:
    """扩展词表 — 添加新 token 并智能初始化 embedding

    策略:
      - "semantic":  用语义最接近的已有 token 均值初始化（最好）
      - "mean":      用全部已有 embedding 的均值（安全）
      - "random":    随机初始化（需要更多训练）

    Args:
        new_tokens:    要添加的 token 列表 (如 ["<think>", "<code>", "领域术语1"])
        new_vocab_size: 目标词表大小 (二选一，和 new_tokens 互斥)
        init_strategy: 初始化策略
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    _safe_update(task, 5, f"加载模型: {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32,
        trust_remote_code=True, device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    old_vocab = len(tokenizer)
    log(f"原始词表: {old_vocab} tokens")

    # 确定新 token
    if new_tokens:
        actually_new = [t for t in new_tokens if t not in tokenizer.get_vocab()]
        if not actually_new:
            raise ValueError("所有 token 已在词表中，无需扩展")
        num_added = tokenizer.add_tokens(actually_new)
        log(f"添加了 {num_added} 个新 token: {actually_new[:10]}...")
    elif new_vocab_size > old_vocab:
        # 生成占位 token
        num_to_add = new_vocab_size - old_vocab
        placeholder_tokens = [f"<extra_{i}>" for i in range(num_to_add)]
        num_added = tokenizer.add_tokens(placeholder_tokens)
        log(f"添加了 {num_added} 个占位 token")
    else:
        raise ValueError("请提供 new_tokens 列表或大于当前词表的 new_vocab_size")

    new_vocab = len(tokenizer)
    _safe_update(task, 20, f"词表: {old_vocab} → {new_vocab}")

    # 调整模型 embedding 大小
    model.resize_token_embeddings(new_vocab)

    # 智能初始化新 embedding
    _safe_update(task, 30, f"初始化新 embedding ({init_strategy})...")

    with torch.no_grad():
        # 找到 input embedding 和 lm_head
        emb_weight = model.get_input_embeddings().weight
        lm_weight = model.get_output_embeddings().weight if model.get_output_embeddings() is not None else None

        if init_strategy == "semantic" and new_tokens:
            # 用语义相近 token 初始化
            for i, token in enumerate(new_tokens[:num_added]):
                idx = old_vocab + i
                # 把新 token 拆成子 token，取其 embedding 均值
                sub_ids = tokenizer.encode(token, add_special_tokens=False)
                sub_ids = [sid for sid in sub_ids if sid < old_vocab]
                if sub_ids:
                    init_vec = emb_weight[sub_ids].mean(dim=0)
                else:
                    init_vec = emb_weight[:old_vocab].mean(dim=0)
                emb_weight[idx] = init_vec
                if lm_weight is not None and lm_weight.shape[0] > idx:
                    lm_weight[idx] = init_vec
        elif init_strategy == "mean":
            mean_vec = emb_weight[:old_vocab].mean(dim=0)
            for idx in range(old_vocab, new_vocab):
                emb_weight[idx] = mean_vec + torch.randn_like(mean_vec) * 0.01
                if lm_weight is not None and lm_weight.shape[0] > idx:
                    lm_weight[idx] = mean_vec + torch.randn_like(mean_vec) * 0.01
        else:  # random — 默认 resize 已是随机，但 scale 可能太大
            std = emb_weight[:old_vocab].std()
            for idx in range(old_vocab, new_vocab):
                emb_weight[idx] = torch.randn_like(emb_weight[0]) * std
                if lm_weight is not None and lm_weight.shape[0] > idx:
                    lm_weight[idx] = torch.randn_like(lm_weight[0]) * std

    _safe_update(task, 70, "保存...")
    out_dir = Path(LORAS_DIR) / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    _copy_custom_model_files(model_path, out_dir)

    _save_expansion_meta(out_dir, model_path, "vocab_expand", {
        "old_vocab": old_vocab, "new_vocab": new_vocab,
        "init_strategy": init_strategy,
        "added_tokens": new_tokens[:20] if new_tokens else f"+{new_vocab - old_vocab} placeholders",
    })

    del model
    gc.collect()
    _safe_update(task, 100, f"✅ 词表扩展: {old_vocab}→{new_vocab} | {out_dir.name}")
    return str(out_dir)


# ════════════════════════════════════════════════════
#  混合膨胀 (同时扩深 + 扩宽)
# ════════════════════════════════════════════════════

def hybrid_expand(
    model_path: str,
    output_name: str,
    target_layers: int = 0,
    target_hidden: int = 0,
    target_intermediate: int = 0,
    depth_strategy: str = "repeat_middle",
    noise_scale: float = 0.01,
    task=None,
) -> str:
    """先扩深再扩宽，一步到位

    例: 12层×768dim (100M) → 24层×1024dim (400M)
    """
    import torch
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    orig_layers = config.num_hidden_layers
    orig_hidden = config.hidden_size

    # 阶段一: 扩深
    if target_layers > orig_layers:
        _safe_update(task, 5, f"阶段一: 深度膨胀 {orig_layers}→{target_layers} 层")
        tmp_name = f"_tmp_depth_{int(time.time())}"
        tmp_path = depth_expand(
            model_path, tmp_name,
            strategy=depth_strategy,
            num_new_layers=target_layers - orig_layers,
            noise_scale=noise_scale,
            task=task,
        )
        stage2_path = tmp_path
    else:
        stage2_path = model_path

    # 阶段二: 扩宽
    if target_hidden > orig_hidden or target_intermediate > 0:
        _safe_update(task, 55, f"阶段二: 宽度膨胀 → hidden={target_hidden}")
        result = width_expand(
            stage2_path, output_name,
            target_hidden=target_hidden,
            target_intermediate=target_intermediate,
            noise_scale=noise_scale,
            task=task,
        )
    else:
        # 只有深度膨胀，重命名
        result = stage2_path
        if stage2_path != model_path:
            import shutil
            final_dir = Path(LORAS_DIR) / output_name
            if final_dir.exists():
                shutil.rmtree(final_dir)
            Path(stage2_path).rename(final_dir)
            result = str(final_dir)

    # 清理临时文件
    if target_layers > orig_layers:
        tmp_dir = Path(LORAS_DIR) / f"_tmp_depth_{int(time.time())}"
        if tmp_dir.exists():
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    _safe_update(task, 100, f"✅ 混合膨胀完成 | {output_name}")
    return result


# ════════════════════════════════════════════════════
#  工具函数
# ════════════════════════════════════════════════════

def _find_layers(model):
    """找到模型的 Transformer 层列表"""
    # 常见路径: model.model.layers / model.transformer.h / model.gpt_neox.layers
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
    raise RuntimeError(f"无法找到模型层。已知路径: {[c[0] for c in candidates]}")


def _set_layers(model, new_layers, attr_path):
    """设置模型的 Transformer 层列表"""
    parts = attr_path.split(".")
    obj = model
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], new_layers)


def _count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _count_params_from_dir(model_path: str) -> int:
    """从 config.json 估算参数量"""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        h = cfg.hidden_size
        n = cfg.num_hidden_layers
        v = cfg.vocab_size
        inter = getattr(cfg, "intermediate_size", h * 4)
        heads = cfg.num_attention_heads
        # 粗估
        emb = v * h
        per_layer = 4 * h * h + 3 * h * inter  # QKV+O + gate/up/down
        return emb + n * per_layer
    except Exception:
        return 0


def _save_expansion_meta(out_dir, source_model, method, details):
    """保存膨胀元信息"""
    meta = {
        "type": "expansion",
        "method": method,
        "source_model": source_model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **details,
    }
    try:
        (Path(out_dir) / "forgex_expansion_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except Exception:
        pass


# ════════════════════════════════════════════════════
#  6. MoE 专家嫁接 — Sparse Upcycling
# ════════════════════════════════════════════════════

def moe_upcycle(
    source_model: str,
    output_name: str,
    num_experts: int = 4,
    top_k: int = 2,
    moe_layers: str = "all",
    noise_scale: float = 0.01,
    task=None,
) -> str:
    """将 Dense FFN 升级为 MoE 稀疏专家层。

    原理 (Google "Sparse Upcycling"):
      - 每个被选中层的 FFN 被复制 N 次, 成为 N 个"专家"
      - 加入可训练 Router 线性层 [hidden → num_experts]
      - 每 token 只经过 top_k 个专家 → FLOPs 不变, 参数量 ×N
      - 后续 CPT 使专家自然分化出不同领域特长

    Args:
        source_model: 源模型 HF ID 或本地路径
        output_name: 输出名
        num_experts: 专家数 (通常 4 或 8)
        top_k: 每 token 激活专家数
        moe_layers: "all" / "even" / "odd" / 逗号分隔层号
        noise_scale: 复制专家时的微扰 (帮助分化)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from safetensors.torch import save_file

    _safe_update(task, 2, f"📏 分析模型: {source_model}")
    config = AutoConfig.from_pretrained(source_model, trust_remote_code=True)
    n_layers = config.num_hidden_layers
    hidden = config.hidden_size
    inter = getattr(config, "intermediate_size", hidden * 4)

    # 解析要转换的层号
    layer_ids = _parse_moe_layer_ids(moe_layers, n_layers)
    if not layer_ids:
        raise ValueError(f"没有有效的 MoE 层 (模型共 {n_layers} 层)")

    _safe_update(task, 5,
        f"🔮 MoE 升级: {len(layer_ids)}/{n_layers} 层 × {num_experts} 专家 (top{top_k})")

    _safe_update(task, 8, "加载模型权重到 CPU...")
    sd = AutoModelForCausalLM.from_pretrained(
        source_model, torch_dtype=torch.float16,
        device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True,
    ).state_dict()
    tokenizer = AutoTokenizer.from_pretrained(source_model, trust_remote_code=True)

    # 检测层前缀和 FFN 键名
    from core.expander import _detect_layer_prefix, _extract_layer_index
    layer_prefix = _detect_layer_prefix(sd, n_layers)
    if not layer_prefix:
        raise ValueError("无法检测层前缀")

    ffn_markers = _detect_ffn_markers(sd, layer_prefix)
    if not ffn_markers:
        raise ValueError("无法检测 FFN 参数模式")

    log(f"层前缀: {layer_prefix}, FFN 标记: {ffn_markers}")

    _safe_update(task, 15, "构建 MoE 权重...")
    new_sd = {}
    orig_total = 0
    new_total = 0

    for key, value in sd.items():
        orig_total += value.numel()
        layer_idx = _extract_layer_index(key, layer_prefix)
        is_moe_layer = layer_idx is not None and layer_idx in layer_ids
        is_ffn_param = is_moe_layer and any(m in key for m in ffn_markers)

        if is_ffn_param:
            # 复制为多个专家
            for e in range(num_experts):
                ek = _to_expert_key(key, e)
                cloned = value.clone()
                if e > 0 and noise_scale > 0 and cloned.is_floating_point():
                    noise = torch.randn_like(cloned) * noise_scale * cloned.abs().mean()
                    cloned = cloned + noise
                new_sd[ek] = cloned
                new_total += cloned.numel()

            # Router 权重 (每层只加一次)
            mlp_base = _get_mlp_base(key, layer_prefix, layer_idx)
            rk = f"{mlp_base}router.weight"
            if rk not in new_sd:
                rw = torch.zeros(num_experts, hidden, dtype=value.dtype)
                torch.nn.init.kaiming_uniform_(rw)
                new_sd[rk] = rw
                new_total += rw.numel()
        else:
            new_sd[key] = value.clone()
            new_total += value.numel()

        # 进度更新
        done_frac = sum(v.numel() for v in new_sd.values()) / max(new_total, 1)
        if int(done_frac * 100) % 10 == 0:
            _safe_update(task, 15 + int(done_frac * 55),
                         f"  构建中: {new_total / 1e6:.0f}M 参数...")

    del sd
    gc.collect()

    _safe_update(task, 75, "保存 MoE 模型...")
    out_dir = Path(LORAS_DIR) / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    config.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    save_file(new_sd, str(out_dir / "model.safetensors"))
    _copy_custom_model_files(source_model, str(out_dir))

    # MoE 配置
    moe_cfg = {
        "num_experts": num_experts,
        "top_k": top_k,
        "moe_layers": layer_ids,
        "ffn_markers": ffn_markers,
        "hidden_size": hidden,
    }
    (out_dir / "forgex_moe_config.json").write_text(
        json.dumps(moe_cfg, indent=2, default=str), encoding="utf-8"
    )

    # 元信息
    _save_expansion_meta(out_dir, source_model, "moe_upcycle", {
        "num_experts": num_experts, "top_k": top_k,
        "moe_layers": layer_ids,
        "orig_params": orig_total, "new_params": new_total,
        "growth": f"{new_total / max(orig_total, 1):.1f}x",
    })

    del new_sd
    gc.collect()

    o_str = f"{orig_total / 1e6:.0f}M" if orig_total < 1e9 else f"{orig_total / 1e9:.1f}B"
    n_str = f"{new_total / 1e6:.0f}M" if new_total < 1e9 else f"{new_total / 1e9:.1f}B"
    log(f"MoE 升级完成: {o_str} → {n_str} ({num_experts}×top{top_k})")
    log(f"  下一步: 用领域语料 CPT 使专家分化")

    _safe_update(task, 100, f"✅ MoE: {o_str} → {n_str}")
    return str(out_dir)


def _parse_moe_layer_ids(spec: str, total: int) -> List[int]:
    """解析 MoE 层规格"""
    spec = spec.strip().lower()
    if spec in ("all", ""):
        return list(range(total))
    if spec == "even":
        return list(range(0, total, 2))
    if spec == "odd":
        return list(range(1, total, 2))
    # 逗号分隔
    return [int(x.strip()) for x in spec.split(",")
            if x.strip().isdigit() and 0 <= int(x.strip()) < total]


def _detect_ffn_markers(sd: Dict, layer_prefix: str) -> List[str]:
    """检测 FFN 参数的键名子串"""
    candidates = [
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        "mlp.fc1", "mlp.fc2", "mlp.dense",
        "mlp.c_fc", "mlp.c_proj",
        "mlp.gate", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h",
        "feed_forward.w1", "feed_forward.w2", "feed_forward.w3",
    ]
    test = f"{layer_prefix}0."
    found = []
    for key in sd:
        if key.startswith(test):
            for pat in candidates:
                if pat in key and pat not in found:
                    found.append(pat)
    return found


def _to_expert_key(key: str, expert_idx: int) -> str:
    """将 FFN key 转为 expert 命名空间"""
    for marker in ["mlp.", "feed_forward.", "ffn."]:
        if marker in key:
            parts = key.split(marker, 1)
            return f"{parts[0]}moe_experts.{expert_idx}.{parts[1]}"
    return key.replace(".weight", f".expert_{expert_idx}.weight")


def _get_mlp_base(key: str, layer_prefix: str, layer_idx: int) -> str:
    """提取 MLP 模块之前的路径前缀"""
    for marker in ["mlp.", "feed_forward.", "ffn."]:
        if marker in key:
            return key[:key.index(marker)]
    return f"{layer_prefix}{layer_idx}."
