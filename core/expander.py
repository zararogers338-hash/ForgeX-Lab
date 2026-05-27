# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

"""ForgeX 模型扩容引擎 — 让模型物理增长以吸收更多知识

三种扩容策略:
  1. 层复制 (Depth Scaling): 复制模型中间层 → 增加深度 → 继续训练使新层分化
     例: 12层 Qwen → 复制 4-8 层 → 16层, 参数 +33%
  2. Frankenmerge (层嫁接): 从不同模型取不同层拼接成一个新模型
     例: 模型A的前8层 + 模型B的后8层 = 混血怪物
  3. 词表扩展: 添加新 token 并用语义相近的已有 token 初始化嵌入
     例: 给英文模型加入 5000 个中文 token

所有操作都是纯权重操作，不需要 GPU，在 CPU 上完成后再训练。
"""

from __future__ import annotations

import copy
import json
import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core import LORAS_DIR, DATASETS_DIR, log


def _safe_update(task, p: float, msg: str):
    if task is not None:
        try:
            task.update_progress(float(p), str(msg))
        except Exception:
            pass


# ════════════════════════════════════════════════════
#  1. 层复制 — Depth Scaling
# ════════════════════════════════════════════════════

def depth_scale(
    source_model: str,
    output_name: str,
    duplicate_range: Tuple[int, int] = (4, 8),
    insert_position: Optional[int] = None,
    add_noise: float = 0.01,
    task=None,
) -> str:
    """复制模型中间层以增加深度。

    原理: Transformer 中间层处理抽象特征，复制后通过少量训练即可分化出新能力。
    研究证明 depth-upscaling 是高效的模型增长方法 (SOLAR 10.7B 就是这样做的)。

    Args:
        source_model: 源模型路径或 HF ID
        output_name: 输出目录名
        duplicate_range: (start, end) 要复制的层范围 (含 start, 不含 end)
        insert_position: 插入位置 (None=原位后面)
        add_noise: 给复制层的权重添加微量噪声，帮助训练时分化 (0=不加)
    
    Returns:
        输出目录路径
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    _safe_update(task, 2, f"📏 加载源模型配置: {source_model}")

    # 加载配置
    config = AutoConfig.from_pretrained(source_model, trust_remote_code=True)
    n_layers = getattr(config, "num_hidden_layers", None)
    if n_layers is None:
        raise ValueError(f"无法读取模型层数: {source_model}")

    start, end = duplicate_range
    if start < 0 or end > n_layers or start >= end:
        raise ValueError(f"层范围无效: ({start}, {end})，模型共 {n_layers} 层")

    n_dup = end - start
    new_n_layers = n_layers + n_dup
    if insert_position is None:
        insert_position = end  # 默认: 在复制范围后面插入

    log(f"层复制: {n_layers}层 → {new_n_layers}层 (复制 L{start}~L{end-1}, 插入位置 L{insert_position})")
    _safe_update(task, 5, f"层复制: {n_layers} → {new_n_layers} 层")

    # 加载模型到 CPU (用 float16 节省 RAM)
    _safe_update(task, 8, "加载源模型权重到 CPU...")
    # 预导入模型类（防止 LazyAutoMapping 失效）
    try:
        from core.safe_loader import ensure_model_importable
        ensure_model_importable(source_model)
    except Exception:
        pass
    model = AutoModelForCausalLM.from_pretrained(
        source_model, torch_dtype=torch.float16,
        device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(source_model, trust_remote_code=True)

    _safe_update(task, 25, "提取权重...")

    # 获取 state_dict
    sd = model.state_dict()

    # 找到层的键名模式 (不同架构可能不同)
    # 常见: model.layers.{i}.xxx 或 transformer.h.{i}.xxx
    layer_prefix = _detect_layer_prefix(sd, n_layers)
    if not layer_prefix:
        raise ValueError("无法检测模型层的键名模式，不支持此架构")

    log(f"层键名前缀: {layer_prefix}")

    # 构建新的 state_dict
    _safe_update(task, 35, f"构建新模型 ({new_n_layers} 层)...")
    new_sd = {}

    for key, value in sd.items():
        # 检查是否是层参数
        layer_idx = _extract_layer_index(key, layer_prefix)

        if layer_idx is None:
            # 非层参数 (embedding, lm_head, norm 等) 直接复制
            new_sd[key] = value.clone()
        else:
            # 层参数: 需要重新编号
            new_idx = _remap_layer_index(
                layer_idx, n_layers, start, end, insert_position
            )
            new_key = key.replace(f"{layer_prefix}{layer_idx}.", f"{layer_prefix}{new_idx}.")
            new_sd[new_key] = value.clone()

    # 添加复制的层
    _safe_update(task, 50, f"复制层 L{start}~L{end-1} → L{insert_position}~L{insert_position + n_dup - 1}...")
    for i in range(n_dup):
        src_layer = start + i
        dst_layer = insert_position + i

        for key, value in sd.items():
            layer_idx = _extract_layer_index(key, layer_prefix)
            if layer_idx == src_layer:
                new_key = key.replace(f"{layer_prefix}{src_layer}.", f"{layer_prefix}{dst_layer}.")
                cloned = value.clone()

                # 添加微量噪声帮助分化
                if add_noise > 0 and cloned.is_floating_point():
                    noise = torch.randn_like(cloned) * add_noise * cloned.abs().mean()
                    cloned = cloned + noise

                new_sd[new_key] = cloned

        pct = 50 + 25 * (i + 1) / n_dup
        _safe_update(task, pct, f"复制层 {i + 1}/{n_dup}")

    # 释放旧模型
    del model, sd
    gc.collect()

    # 更新配置
    _safe_update(task, 78, "保存扩容后模型...")
    config.num_hidden_layers = new_n_layers

    # 保存
    out_dir = Path(LORAS_DIR) / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    config.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # 保存权重
    from safetensors.torch import save_file
    save_file(new_sd, str(out_dir / "model.safetensors"))

    del new_sd
    gc.collect()

    # 保存元信息
    old_params = _count_params_from_config(config, n_layers)
    new_params = _count_params_from_config(config, new_n_layers)
    meta = {
        "type": "expanded",
        "method": "depth_scale",
        "source_model": source_model,
        "original_layers": n_layers,
        "new_layers": new_n_layers,
        "duplicate_range": [start, end],
        "insert_position": insert_position,
        "noise": add_noise,
        "original_params_approx": old_params,
        "new_params_approx": new_params,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "forgex_expand_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _safe_update(task, 95, f"✅ {n_layers}层 → {new_n_layers}层 | 参数: ~{old_params / 1e6:.0f}M → ~{new_params / 1e6:.0f}M")
    log(f"模型扩容完成: {out_dir}")
    log(f"  层数: {n_layers} → {new_n_layers} (+{n_dup})")
    log(f"  参数: ~{old_params / 1e6:.0f}M → ~{new_params / 1e6:.0f}M (+{(new_params - old_params) / 1e6:.0f}M)")
    log(f"  下一步: 用新语料在扩容后的模型上继续预训练 (CPT)，使新层学到新知识")

    _safe_update(task, 100, f"✅ 完成: {out_dir.name}")
    return str(out_dir)


# ════════════════════════════════════════════════════
#  2. Frankenmerge — 层嫁接
# ════════════════════════════════════════════════════

def frankenmerge(
    layer_specs: List[Dict],
    output_name: str,
    tokenizer_source: Optional[str] = None,
    task=None,
) -> str:
    """从多个模型的不同层范围拼接成一个新模型。

    这就是社区所说的 "Frankenmerge" 或 "Passthrough merge"。
    经典用法: 模型A擅长推理(中间层强) + 模型B擅长生成(首尾层强) → 拼接。

    Args:
        layer_specs: 层规格列表，如:
            [
                {"model": "模型A路径", "start": 0, "end": 8},
                {"model": "模型B路径", "start": 8, "end": 16},
            ]
        output_name: 输出名
        tokenizer_source: tokenizer 来源模型 (None=用第一个模型的)

    注意: 所有模型必须同架构、同 hidden_size。
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from safetensors.torch import save_file

    if not layer_specs:
        raise ValueError("至少需要一个层规格")

    _safe_update(task, 2, "验证模型兼容性...")

    # 验证所有模型兼容
    configs = []
    hidden_sizes = set()
    for spec in layer_specs:
        cfg = AutoConfig.from_pretrained(spec["model"], trust_remote_code=True)
        configs.append(cfg)
        hs = getattr(cfg, "hidden_size", None)
        if hs:
            hidden_sizes.add(hs)

    if len(hidden_sizes) > 1:
        raise ValueError(f"模型 hidden_size 不一致: {hidden_sizes}。Frankenmerge 要求所有模型同架构同大小。")

    # 计算新模型总层数
    total_new_layers = sum(spec["end"] - spec["start"] for spec in layer_specs)
    log(f"Frankenmerge: {len(layer_specs)} 个片段 → {total_new_layers} 层")

    _safe_update(task, 5, f"拼接 {total_new_layers} 层...")

    new_sd = {}
    current_dst_layer = 0
    non_layer_sd = None  # 非层参数 (从第一个模型取)
    layer_prefix = None

    for seg_idx, spec in enumerate(layer_specs):
        model_path = spec["model"]
        src_start = spec["start"]
        src_end = spec["end"]
        n_seg = src_end - src_start

        pct_base = 5 + 80 * seg_idx / len(layer_specs)
        _safe_update(task, pct_base, f"加载片段 {seg_idx + 1}/{len(layer_specs)}: {Path(model_path).name} L{src_start}~L{src_end - 1}")

        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16,
            device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True,
        )
        sd = model.state_dict()

        if layer_prefix is None:
            n_layers = getattr(configs[0], "num_hidden_layers", 32)
            layer_prefix = _detect_layer_prefix(sd, n_layers)
            if not layer_prefix:
                raise ValueError(f"无法检测层前缀: {model_path}")

        # 非层参数只取一次
        if non_layer_sd is None:
            for key, value in sd.items():
                idx = _extract_layer_index(key, layer_prefix)
                if idx is None:
                    non_layer_sd = non_layer_sd or {}
                    non_layer_sd[key] = value.clone()

        # 复制指定范围的层
        for src_layer in range(src_start, src_end):
            dst_layer = current_dst_layer + (src_layer - src_start)
            for key, value in sd.items():
                idx = _extract_layer_index(key, layer_prefix)
                if idx == src_layer:
                    new_key = key.replace(f"{layer_prefix}{src_layer}.", f"{layer_prefix}{dst_layer}.")
                    new_sd[new_key] = value.clone()

        current_dst_layer += n_seg

        del model, sd
        gc.collect()

    # 合并非层参数
    if non_layer_sd:
        new_sd.update(non_layer_sd)

    _safe_update(task, 88, "保存 Frankenmerge 模型...")

    out_dir = Path(LORAS_DIR) / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 更新配置
    new_config = copy.deepcopy(configs[0])
    new_config.num_hidden_layers = total_new_layers
    new_config.save_pretrained(str(out_dir))

    # Tokenizer
    tok_src = tokenizer_source or layer_specs[0]["model"]
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    tokenizer.save_pretrained(str(out_dir))

    # 保存权重
    save_file(new_sd, str(out_dir / "model.safetensors"))

    del new_sd
    gc.collect()

    # 元信息
    meta = {
        "type": "expanded",
        "method": "frankenmerge",
        "segments": [
            {"model": s["model"], "start": s["start"], "end": s["end"]}
            for s in layer_specs
        ],
        "total_layers": total_new_layers,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "forgex_expand_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log(f"Frankenmerge 完成: {total_new_layers} 层 → {out_dir}")
    _safe_update(task, 100, f"✅ Frankenmerge: {total_new_layers} 层")
    return str(out_dir)


# ════════════════════════════════════════════════════
#  3. 词表扩展
# ════════════════════════════════════════════════════

def expand_vocabulary(
    source_model: str,
    output_name: str,
    new_tokens: List[str],
    init_method: str = "mean",
    task=None,
) -> str:
    """给模型添加新 token 并智能初始化嵌入向量。

    初始化方法:
      - "mean": 用所有已有 token 嵌入的均值 (安全默认)
      - "similar": 对每个新 token，找语义最相近的已有 token 来初始化
      - "random": 随机初始化 (需要更多训练)

    Args:
        source_model: 源模型
        output_name: 输出名
        new_tokens: 要添加的 token 列表
        init_method: 初始化方法
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not new_tokens:
        raise ValueError("请提供要添加的 token 列表")

    _safe_update(task, 5, f"加载模型: {source_model}")

    model = AutoModelForCausalLM.from_pretrained(
        source_model, torch_dtype=torch.float16,
        device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(source_model, trust_remote_code=True)

    old_vocab_size = len(tokenizer)

    _safe_update(task, 20, f"添加 {len(new_tokens)} 个新 token...")

    # 过滤已存在的 token
    existing = set(tokenizer.get_vocab().keys())
    truly_new = [t for t in new_tokens if t not in existing]
    if not truly_new:
        raise ValueError("所有提供的 token 都已存在于词表中")

    log(f"词表扩展: {old_vocab_size} → {old_vocab_size + len(truly_new)} (+{len(truly_new)})")

    # 添加 tokens
    tokenizer.add_tokens(truly_new)
    model.resize_token_embeddings(len(tokenizer))

    _safe_update(task, 40, f"初始化新嵌入 ({init_method})...")

    # 智能初始化
    embed_weight = model.get_input_embeddings().weight.data
    lm_head_weight = model.get_output_embeddings().weight.data if model.get_output_embeddings() is not None else None

    if init_method == "mean":
        # 用已有 token 的均值初始化
        mean_embed = embed_weight[:old_vocab_size].mean(dim=0)
        for i in range(old_vocab_size, len(tokenizer)):
            embed_weight[i] = mean_embed + torch.randn_like(mean_embed) * 0.01
            if lm_head_weight is not None and i < lm_head_weight.shape[0]:
                lm_head_weight[i] = mean_embed + torch.randn_like(mean_embed) * 0.01

    elif init_method == "similar":
        # 对每个新 token，找最相似的已有 token
        # 用 token 字符串的子串匹配作为启发式
        for j, token in enumerate(truly_new):
            idx = old_vocab_size + j
            # 尝试找子串匹配
            best_match = _find_similar_token(token, tokenizer, old_vocab_size)
            if best_match is not None:
                embed_weight[idx] = embed_weight[best_match].clone() + torch.randn_like(embed_weight[best_match]) * 0.01
                if lm_head_weight is not None and idx < lm_head_weight.shape[0]:
                    lm_head_weight[idx] = lm_head_weight[best_match].clone() + torch.randn_like(lm_head_weight[best_match]) * 0.01
            else:
                mean_embed = embed_weight[:old_vocab_size].mean(dim=0)
                embed_weight[idx] = mean_embed + torch.randn_like(mean_embed) * 0.02

            if (j + 1) % 100 == 0:
                _safe_update(task, 40 + 30 * (j + 1) / len(truly_new),
                             f"初始化嵌入: {j + 1}/{len(truly_new)}")

    else:  # random
        # Xavier/正态分布初始化
        std = embed_weight[:old_vocab_size].std().item()
        for i in range(old_vocab_size, len(tokenizer)):
            embed_weight[i] = torch.randn_like(embed_weight[i]) * std

    _safe_update(task, 75, "保存扩展后模型...")

    out_dir = Path(LORAS_DIR) / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))

    # 元信息
    meta = {
        "type": "expanded",
        "method": "vocab_expand",
        "source_model": source_model,
        "old_vocab_size": old_vocab_size,
        "new_vocab_size": len(tokenizer),
        "tokens_added": len(truly_new),
        "init_method": init_method,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "forgex_expand_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    del model
    gc.collect()

    log(f"词表扩展完成: {old_vocab_size} → {old_vocab_size + len(truly_new)}")
    _safe_update(task, 100, f"✅ 词表: {old_vocab_size} → {old_vocab_size + len(truly_new)}")
    return str(out_dir)


# ════════════════════════════════════════════════════
#  辅助函数
# ════════════════════════════════════════════════════

def _detect_layer_prefix(state_dict: Dict, n_layers: int) -> Optional[str]:
    """自动检测层键名前缀。

    支持:
      - model.layers.{i}.xxx  (LLaMA, Qwen, Mistral, ...)
      - transformer.h.{i}.xxx (GPT-2, GPT-J)
      - transformer.layers.{i}.xxx
      - gpt_neox.layers.{i}.xxx
    """
    patterns = [
        "model.layers.",
        "transformer.h.",
        "transformer.layers.",
        "gpt_neox.layers.",
        "model.decoder.layers.",
    ]
    for prefix in patterns:
        # 检查是否存在 prefix + "0."
        key_test = f"{prefix}0."
        if any(k.startswith(key_test) for k in state_dict):
            return prefix
    return None


def _extract_layer_index(key: str, prefix: str) -> Optional[int]:
    """从键名中提取层索引。"""
    if prefix not in key:
        return None
    try:
        after = key[key.index(prefix) + len(prefix):]
        idx_str = after.split(".")[0]
        return int(idx_str)
    except (ValueError, IndexError):
        return None


def _remap_layer_index(
    old_idx: int,
    n_old: int,
    dup_start: int,
    dup_end: int,
    insert_pos: int,
) -> int:
    """重新映射原始层索引到新模型中的位置。

    层复制后的布局:
      原始: [0, 1, ..., insert_pos-1, insert_pos, ..., n_old-1]
      新的: [0, 1, ..., insert_pos-1, DUP_0, DUP_1, ..., DUP_k, insert_pos, ..., n_old-1]
    """
    n_dup = dup_end - dup_start
    if old_idx < insert_pos:
        return old_idx
    else:
        return old_idx + n_dup


def _count_params_from_config(config, n_layers: int) -> int:
    """根据配置粗估参数量。"""
    h = getattr(config, "hidden_size", 768)
    v = getattr(config, "vocab_size", 32000)
    i = getattr(config, "intermediate_size", h * 4)
    n_heads = getattr(config, "num_attention_heads", 12)
    kv_heads = getattr(config, "num_key_value_heads", n_heads)

    # Embedding
    emb = v * h
    # Per-layer
    qkv = h * (h + 2 * (h * kv_heads // n_heads))
    o = h * h
    mlp = h * i * 3  # gate + up + down
    norms = h * 2
    per_layer = qkv + o + mlp + norms
    return emb + n_layers * per_layer + h


def _find_similar_token(new_token: str, tokenizer, old_vocab_size: int) -> Optional[int]:
    """用字符串匹配找最相似的已有 token。"""
    vocab = tokenizer.get_vocab()
    best_overlap = 0
    best_idx = None

    # 尝试完全匹配子串
    for existing_token, idx in vocab.items():
        if idx >= old_vocab_size:
            continue
        # 计算字符重叠
        overlap = 0
        min_len = min(len(new_token), len(existing_token))
        for i in range(min_len):
            if new_token[i] == existing_token[i]:
                overlap += 1
            else:
                break
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = idx

    # 至少要有 2 个字符匹配才算
    return best_idx if best_overlap >= 2 else None


def analyze_model_for_expansion(model_path: str) -> Dict:
    """分析模型，给出扩容建议。"""
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        return {"error": str(e)}

    n_layers = getattr(cfg, "num_hidden_layers", 0)
    hidden = getattr(cfg, "hidden_size", 0)
    vocab = getattr(cfg, "vocab_size", 0)
    inter = getattr(cfg, "intermediate_size", 0)
    n_heads = getattr(cfg, "num_attention_heads", 0)
    kv_heads = getattr(cfg, "num_key_value_heads", n_heads)

    params = _count_params_from_config(cfg, n_layers)

    # 推荐的层复制范围
    dup_start = n_layers // 3
    dup_end = n_layers * 2 // 3
    new_params = _count_params_from_config(cfg, n_layers + (dup_end - dup_start))

    # MoE 估算
    ffn_per_layer = hidden * inter * 3  # gate+up+down
    moe4_params = params + ffn_per_layer * n_layers * 3  # 4 experts: +3x FFN
    moe8_params = params + ffn_per_layer * n_layers * 7  # 8 experts: +7x FFN

    return {
        "model": model_path,
        "layers": n_layers,
        "hidden_size": hidden,
        "vocab_size": vocab,
        "intermediate_size": inter,
        "heads": n_heads,
        "kv_heads": kv_heads,
        "params": params,
        "params_str": f"{params / 1e6:.0f}M" if params < 1e9 else f"{params / 1e9:.1f}B",
        "recommend": {
            "dup_start": dup_start,
            "dup_end": dup_end,
            "new_layers": n_layers + (dup_end - dup_start),
            "new_params": new_params,
            "new_params_str": f"{new_params / 1e6:.0f}M" if new_params < 1e9 else f"{new_params / 1e9:.1f}B",
            "ram_needed_gb": round(params * 2 * 2 / 1e9, 1),
        },
        "moe_estimate": {
            "4_experts": f"{moe4_params / 1e6:.0f}M" if moe4_params < 1e9 else f"{moe4_params / 1e9:.1f}B",
            "8_experts": f"{moe8_params / 1e6:.0f}M" if moe8_params < 1e9 else f"{moe8_params / 1e9:.1f}B",
        },
    }


# ════════════════════════════════════════════════════
#  4. MoE 专家嫁接 — Sparse Upcycling
# ════════════════════════════════════════════════════

def sparse_upcycle(
    source_model: str,
    output_name: str,
    num_experts: int = 4,
    top_k: int = 2,
    moe_layer_indices: Optional[List[int]] = None,
    noise_scale: float = 0.01,
    task=None,
) -> str:
    """将 Dense FFN 升级为 MoE 稀疏专家层。

    原理 (Google "Sparse Upcycling" 论文):
      - 每个被选中的 transformer 层的 FFN 被复制 N 次，成为 N 个"专家"
      - 加入一个可训练的 router 线性层 [hidden → num_experts]
      - 每个 token 只经过 top_k 个专家 → 推理 FLOPs 不变
      - 参数量: FFN 部分 ×N → 知识容量大幅增加

    好处:
      - 模型容量翻倍但推理速度几乎不变
      - 每个专家可以分化出不同的领域特长
      - 后续用不同领域语料 CPT，专家自然分化

    Args:
        source_model: 源模型路径或 HF ID
        output_name: 输出名称
        num_experts: 专家数 (通常 4/8)
        top_k: 每 token 激活的专家数 (通常 1 或 2)
        moe_layer_indices: 要转为 MoE 的层号 (None=全部)
        noise_scale: 专家初始化微扰幅度
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from safetensors.torch import save_file

    _safe_update(task, 2, f"📏 分析模型: {source_model}")
    config = AutoConfig.from_pretrained(source_model, trust_remote_code=True)
    n_layers = config.num_hidden_layers
    hidden = config.hidden_size
    inter = getattr(config, "intermediate_size", hidden * 4)

    if moe_layer_indices is None:
        moe_layer_indices = list(range(n_layers))

    moe_layer_indices = [i for i in moe_layer_indices if 0 <= i < n_layers]
    if not moe_layer_indices:
        raise ValueError("没有有效的 MoE 层索引")

    _safe_update(task, 5,
        f"🔮 MoE 升级: {len(moe_layer_indices)}/{n_layers} 层 × {num_experts} 专家 (top{top_k})")

    _safe_update(task, 8, "加载模型权重到 CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        source_model, torch_dtype=torch.float16,
        device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(source_model, trust_remote_code=True)

    sd = model.state_dict()
    layer_prefix = _detect_layer_prefix(sd, n_layers)
    if not layer_prefix:
        raise ValueError("无法检测层前缀")

    _safe_update(task, 20, "构建 MoE 权重...")

    # 找出 FFN 相关的键名模式
    # 通常: mlp.gate_proj, mlp.up_proj, mlp.down_proj (Llama/Qwen)
    #   或: mlp.fc1, mlp.fc2 (GPT-2)
    ffn_patterns = _detect_ffn_key_patterns(sd, layer_prefix)
    if not ffn_patterns:
        raise ValueError("无法检测 FFN 键名模式，不支持此架构")

    log(f"FFN 键名模式: {ffn_patterns}")

    new_sd = {}
    orig_params = 0
    new_params = 0

    for key, value in sd.items():
        layer_idx = _extract_layer_index(key, layer_prefix)
        orig_params += value.numel()

        if layer_idx is not None and layer_idx in moe_layer_indices:
            # 检查是否是 FFN 参数
            is_ffn = any(pat in key for pat in ffn_patterns)

            if is_ffn:
                # 复制为多个专家
                for e in range(num_experts):
                    expert_key = _make_expert_key(key, e)
                    cloned = value.clone()
                    # 第 0 号专家保持原样，其他加噪声促进分化
                    if e > 0 and noise_scale > 0 and cloned.is_floating_point():
                        noise = torch.randn_like(cloned) * noise_scale * cloned.abs().mean()
                        cloned = cloned + noise
                    new_sd[expert_key] = cloned
                    new_params += cloned.numel()

                # 添加 router 权重
                # 从 key 中提取 MLP 部分的前缀
                mlp_prefix = _extract_mlp_prefix(key, layer_prefix, layer_idx)
                router_key = f"{mlp_prefix}router.weight"
                if router_key not in new_sd:
                    router_w = torch.zeros(num_experts, hidden, dtype=value.dtype)
                    torch.nn.init.kaiming_uniform_(router_w)
                    new_sd[router_key] = router_w
                    new_params += router_w.numel()
            else:
                new_sd[key] = value.clone()
                new_params += value.numel()
        else:
            new_sd[key] = value.clone()
            new_params += value.numel()

        if (orig_params // 10_000_000) != ((orig_params - value.numel()) // 10_000_000):
            pct = 20 + 50 * orig_params / sum(v.numel() for v in sd.values())
            _safe_update(task, min(70, pct), f"  处理中... {orig_params / 1e6:.0f}M params")

    del model, sd
    gc.collect()

    _safe_update(task, 75, "保存 MoE 模型...")
    out_dir = Path(LORAS_DIR) / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置 (添加 MoE 元数据到 config，便于后续加载)
    config.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # 保存 MoE 权重
    save_file(new_sd, str(out_dir / "model.safetensors"))

    del new_sd
    gc.collect()

    # 元信息
    meta = {
        "type": "expanded",
        "method": "sparse_upcycle",
        "source_model": source_model,
        "num_experts": num_experts,
        "top_k": top_k,
        "moe_layers": moe_layer_indices,
        "noise_scale": noise_scale,
        "original_params": orig_params,
        "new_params": new_params,
        "ffn_patterns": ffn_patterns,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "forgex_expand_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 保存一个 MoE 配置文件 (用于自定义前向推理)
    moe_config = {
        "num_experts": num_experts,
        "top_k": top_k,
        "moe_layers": moe_layer_indices,
        "ffn_patterns": ffn_patterns,
        "hidden_size": hidden,
    }
    (out_dir / "forgex_moe_config.json").write_text(
        json.dumps(moe_config, indent=2), encoding="utf-8"
    )

    o_str = f"{orig_params / 1e6:.0f}M" if orig_params < 1e9 else f"{orig_params / 1e9:.1f}B"
    n_str = f"{new_params / 1e6:.0f}M" if new_params < 1e9 else f"{new_params / 1e9:.1f}B"
    log(f"MoE 升级完成: {out_dir}")
    log(f"  参数: {o_str} → {n_str} ({num_experts} experts × top{top_k})")
    log(f"  MoE 层: {len(moe_layer_indices)}/{n_layers}")
    log(f"  下一步: 用领域语料 CPT 训练，使专家分化")

    _safe_update(task, 100, f"✅ MoE: {o_str} → {n_str} ({num_experts}×top{top_k})")
    return str(out_dir)


def _detect_ffn_key_patterns(sd: Dict, layer_prefix: str) -> List[str]:
    """检测 FFN 相关参数的键名模式"""
    # 常见 FFN 键名子串
    candidates = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                   "mlp.fc1", "mlp.fc2", "mlp.dense",
                   "feed_forward.w1", "feed_forward.w2", "feed_forward.w3",
                   "mlp.c_fc", "mlp.c_proj",
                   "mlp.gate", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]

    # 只检查第 0 层
    test_prefix = f"{layer_prefix}0."
    found = []
    for key in sd:
        if key.startswith(test_prefix):
            for pat in candidates:
                if pat in key and pat not in found:
                    found.append(pat)
    return found


def _make_expert_key(orig_key: str, expert_idx: int) -> str:
    """将原始 FFN key 转为 expert_{i} 命名空间

    例: model.layers.0.mlp.gate_proj.weight
      → model.layers.0.moe_experts.0.gate_proj.weight
    """
    # 找到 "mlp." 或 "feed_forward." 并替换
    for marker in ["mlp.", "feed_forward.", "ffn."]:
        if marker in orig_key:
            parts = orig_key.split(marker, 1)
            return f"{parts[0]}moe_experts.{expert_idx}.{parts[1]}"
    # fallback: 在 weight 前加 expert 前缀
    return orig_key.replace(".weight", f".expert_{expert_idx}.weight").replace(
        ".bias", f".expert_{expert_idx}.bias"
    )


def _extract_mlp_prefix(key: str, layer_prefix: str, layer_idx: int) -> str:
    """提取到 MLP 模块的完整前缀

    例: model.layers.5.mlp.gate_proj.weight → model.layers.5.
    """
    for marker in ["mlp.", "feed_forward.", "ffn."]:
        if marker in key:
            return key[:key.index(marker)]
    return f"{layer_prefix}{layer_idx}."
