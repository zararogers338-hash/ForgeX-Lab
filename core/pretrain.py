# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

"""ForgeX 从零定制引擎 — 构建 + 预训练 + 继续预训练。

两种核心模式:
  A) 从零构建: 选择架构 → 训练 tokenizer → 从随机权重预训练
  B) 继续预训练 (CPT): 在现有模型上，用新领域语料继续全参数训练

设计约束:
  - 8GB VRAM (RTX 5060) 安全运行 ≤ 350M 参数模型
  - 自动 VRAM 估算，超限自动降级 batch / 启用 gradient checkpointing
  - 兼容 Blackwell GPU (compute cap ≥ 10.0): 使用 SDPA/eager
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from core import DATASETS_DIR, LORAS_DIR, MODELS_CACHE_DIR, log


# ═══ 早期 GPU 兼容性检查（与 trainer.py 相同）═══
def _early_gpu_compat_check():
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            if cap[0] >= 10:
                import os
                os.environ["XFORMERS_DISABLED"] = "1"
                try:
                    import transformers.utils
                    transformers.utils.is_xformers_available = lambda: False
                except Exception:
                    pass
    except Exception:
        pass

_early_gpu_compat_check()

# ════════════════════════════════════════════════════
#  架构预设 — 基于 LLaMA 架构 (RMSNorm, SwiGLU, RoPE)
# ════════════════════════════════════════════════════

ARCH_PRESETS = {
    "nano-25M": {
        "label": "Nano (~25M) — 学习实验，几分钟训完",
        "hidden_size": 384,
        "num_hidden_layers": 6,
        "num_attention_heads": 6,
        "num_key_value_heads": 6,
        "intermediate_size": 1024,
        "max_position_embeddings": 2048,
        "cpu_offload": False,
    },
    "micro-40M": {
        "label": "Micro (~40M) — 轻量任务，30分钟可用",
        "hidden_size": 512,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "intermediate_size": 1408,
        "max_position_embeddings": 2048,
        "cpu_offload": False,
    },
    "mini-100M": {
        "label": "Mini (~100M) — GPT-2 级别，平衡之选",
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 4,
        "intermediate_size": 2048,
        "max_position_embeddings": 2048,
        "cpu_offload": False,
    },
    "small-300M": {
        "label": "Small (~300M) — GPT-2 Medium 级",
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "intermediate_size": 2816,
        "max_position_embeddings": 4096,
        "cpu_offload": False,
    },
    "medium-500M": {
        "label": "Medium (~500M) — CPU offload，训练慢但更强",
        "hidden_size": 1280,
        "num_hidden_layers": 28,
        "num_attention_heads": 20,
        "num_key_value_heads": 4,
        "intermediate_size": 3456,
        "max_position_embeddings": 4096,
        "cpu_offload": True,
    },
    "large-1B": {
        "label": "Large (~1B) — CPU offload + 梯度检查点，需耐心",
        "hidden_size": 2048,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "intermediate_size": 5632,
        "max_position_embeddings": 4096,
        "cpu_offload": True,
    },
}


@dataclass
class ArchConfig:
    """模型架构配置"""
    preset: str = "mini-100M"
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    intermediate_size: int = 2048
    max_position_embeddings: int = 2048
    vocab_size: int = 32000
    cpu_offload: bool = False

    @classmethod
    def from_preset(cls, preset: str, vocab_size: int = 32000) -> "ArchConfig":
        p = ARCH_PRESETS.get(preset, ARCH_PRESETS["mini-100M"])
        return cls(
            preset=preset,
            hidden_size=p["hidden_size"],
            num_hidden_layers=p["num_hidden_layers"],
            num_attention_heads=p["num_attention_heads"],
            num_key_value_heads=p["num_key_value_heads"],
            intermediate_size=p["intermediate_size"],
            max_position_embeddings=p["max_position_embeddings"],
            vocab_size=vocab_size,
            cpu_offload=p.get("cpu_offload", False),
        )

    def param_count(self) -> int:
        """估算总参数量"""
        h = self.hidden_size
        n = self.num_hidden_layers
        i = self.intermediate_size
        v = self.vocab_size
        # Embedding
        emb = v * h
        # Per layer: attention (QKV + O) + MLP (gate + up + down) + norms
        kv_h = self.num_key_value_heads
        attn_h = self.num_attention_heads
        head_dim = h // attn_h
        qkv = h * (attn_h + 2 * kv_h) * head_dim
        o_proj = h * h
        mlp = h * i * 3  # gate + up + down (SwiGLU)
        norms = h * 2  # RMSNorm x2
        per_layer = qkv + o_proj + mlp + norms
        # LM head (tied with embedding in most configs)
        lm_head = 0  # tied
        total = emb + n * per_layer + lm_head + h  # final norm
        return total

    def param_count_str(self) -> str:
        c = self.param_count()
        if c >= 1e9:
            return f"{c / 1e9:.1f}B"
        return f"{c / 1e6:.0f}M"

    def estimate_vram_mb(self, batch_size: int = 1, seq_len: int = 512,
                          grad_ckpt: bool = True) -> int:
        """估算训练 VRAM (MB)"""
        params = self.param_count()
        # 模型参数 (bf16 = 2 bytes)
        model_mb = params * 2 / 1024 / 1024

        if self.cpu_offload:
            # CPU offload 模式: 优化器状态大部分在 CPU，GPU 只保留当前页
            optim_mb = params * 0.5 / 1024 / 1024  # ~25% 常驻 GPU
            grad_mb = params * 2 / 1024 / 1024      # 梯度仍在 GPU
            act_per_layer = batch_size * seq_len * self.hidden_size * 2 / 1024 / 1024
            act_mb = act_per_layer * 2  # grad_ckpt 强制开启
            total = model_mb + optim_mb + grad_mb + act_mb + 500
        else:
            # 标准模式
            optim_mb = params * 2 / 1024 / 1024  # paged_adamw_8bit
            grad_mb = params * 2 / 1024 / 1024
            act_per_layer = batch_size * seq_len * self.hidden_size * 2 / 1024 / 1024
            if grad_ckpt:
                act_mb = act_per_layer * 2
            else:
                act_mb = act_per_layer * self.num_hidden_layers
            total = model_mb + optim_mb + grad_mb + act_mb + 500

        return int(total)


def _safe_update(task, p: float, msg: str):
    if task is not None:
        try:
            task.update_progress(float(p), str(msg))
        except Exception:
            pass


def _detect_gpu() -> Dict[str, Any]:
    info = {"cuda": False, "bf16": False, "vram_mb": 0, "compute_cap": (0, 0)}
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda"] = True
            cap = torch.cuda.get_device_capability()
            info["compute_cap"] = cap
            info["bf16"] = cap[0] >= 8
            info["vram_mb"] = torch.cuda.get_device_properties(0).total_mem // (1024 * 1024)
    except Exception:
        pass
    return info


# ════════════════════════════════════════════════════
#  Tokenizer 训练
# ════════════════════════════════════════════════════

def train_tokenizer(
    corpus_paths: List[Path],
    vocab_size: int = 32000,
    output_dir: Path = None,
    task=None,
) -> str:
    """从语料训练 BPE tokenizer。

    使用 HuggingFace tokenizers 库 (Rust 实现，极快)。
    输出兼容 transformers 的 tokenizer 文件。

    Returns:
        输出目录路径
    """
    _safe_update(task, 5, f"训练 Tokenizer (vocab={vocab_size})...")

    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
    except ImportError:
        raise RuntimeError("需要安装 tokenizers: pip install tokenizers")

    if output_dir is None:
        output_dir = MODELS_CACHE_DIR / f"tokenizer_v{vocab_size}_{int(time.time())}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 构建 BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # 特殊 tokens
    special_tokens = [
        "<unk>", "<s>", "</s>", "<pad>",
        "<|im_start|>", "<|im_end|>",
        "<|system|>", "<|user|>", "<|assistant|>",
    ]

    trainer_obj = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        show_progress=True,
    )

    # 收集语料文本
    _safe_update(task, 10, "读取训练语料...")
    text_files = []
    for p in corpus_paths:
        p = Path(p)
        if p.suffix in (".txt", ".md"):
            text_files.append(str(p))
        elif p.suffix in (".jsonl", ".json"):
            # 提取 text 字段到临时文件
            tmp = output_dir / f"_corpus_{p.stem}.txt"
            lines = []
            raw = p.read_text(encoding="utf-8-sig")
            if p.suffix == ".jsonl":
                for line in raw.strip().splitlines():
                    try:
                        obj = json.loads(line)
                        text = _extract_text(obj)
                        if text:
                            lines.append(text)
                    except Exception:
                        continue
            else:
                try:
                    data = json.loads(raw)
                    if isinstance(data, list):
                        for obj in data:
                            text = _extract_text(obj)
                            if text:
                                lines.append(text)
                except Exception:
                    pass
            if lines:
                tmp.write_text("\n".join(lines), encoding="utf-8")
                text_files.append(str(tmp))
        else:
            # 尝试当纯文本读
            try:
                content = p.read_text(encoding="utf-8-sig")
                if len(content.strip()) > 100:
                    text_files.append(str(p))
            except Exception:
                pass

    if not text_files:
        raise RuntimeError("没有找到可用的语料文件。支持 .txt, .md, .jsonl, .json")

    log(f"Tokenizer 训练语料: {len(text_files)} 个文件")
    _safe_update(task, 15, f"训练 BPE tokenizer ({len(text_files)} 文件, vocab={vocab_size})...")

    tokenizer.train(text_files, trainer_obj)

    # 后处理：添加 chat template 支持
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # 保存为 HuggingFace 格式
    _safe_update(task, 25, "保存 tokenizer...")
    tokenizer.save(str(output_dir / "tokenizer.json"))

    # 创建 tokenizer_config.json
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "model_max_length": 2048,
        "chat_template": (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
            "{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        ),
    }
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    # 创建 special_tokens_map.json
    st_map = {
        "bos_token": "<s>", "eos_token": "</s>",
        "unk_token": "<unk>", "pad_token": "<pad>",
    }
    (output_dir / "special_tokens_map.json").write_text(
        json.dumps(st_map, indent=2, default=str), encoding="utf-8"
    )

    vocab_actual = tokenizer.get_vocab_size()
    _safe_update(task, 30, f"✅ Tokenizer 训练完成: vocab={vocab_actual}")
    log(f"Tokenizer 保存到: {output_dir} (vocab={vocab_actual})")
    return str(output_dir)


def _extract_text(obj) -> str:
    """从 JSON 对象提取文本"""
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, dict):
        return ""
    # 常见字段
    for k in ("text", "content", "instruction", "output", "response", "question", "answer"):
        v = obj.get(k)
        if v and isinstance(v, str):
            return v
    # messages 格式
    msgs = obj.get("messages") or obj.get("conversations")
    if isinstance(msgs, list):
        parts = []
        for m in msgs:
            if isinstance(m, dict):
                c = m.get("content") or m.get("value") or ""
                if c:
                    parts.append(str(c))
        return "\n".join(parts)
    return ""


# ════════════════════════════════════════════════════
#  模型构建
# ════════════════════════════════════════════════════

def build_model(
    arch: ArchConfig,
    tokenizer_path: str,
    output_dir: str = None,
    task=None,
):
    """从架构配置构建全新的 LLaMA 模型。

    Returns:
        (model, tokenizer, output_dir)
    """
    _safe_update(task, 30, f"构建模型: {arch.param_count_str()} ({arch.preset})...")

    import torch
    # 确保兼容层已激活（torchvision 防护 + transformers stubs）
    try:
        from core.safe_loader import _patch_transformers_compat, _ensure_torchvision_safe
        _ensure_torchvision_safe()
        _patch_transformers_compat()
    except Exception:
        pass
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    # 加载 tokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    actual_vocab = max(tok.vocab_size, len(tok))

    # 构建 config
    # vocab_size 对齐到 64 的倍数（优化矩阵运算）
    vocab_aligned = ((actual_vocab + 63) // 64) * 64
    arch.vocab_size = vocab_aligned

    config = LlamaConfig(
        vocab_size=vocab_aligned,
        hidden_size=arch.hidden_size,
        intermediate_size=arch.intermediate_size,
        num_hidden_layers=arch.num_hidden_layers,
        num_attention_heads=arch.num_attention_heads,
        num_key_value_heads=arch.num_key_value_heads,
        max_position_embeddings=arch.max_position_embeddings,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
        use_cache=False,
        bos_token_id=tok.bos_token_id or 1,
        eos_token_id=tok.eos_token_id or 2,
        pad_token_id=tok.pad_token_id or 0,
    )

    # 尝试 SDPA → eager fallback（Blackwell 兼容）
    gpu = _detect_gpu()
    attn_impl = None
    if gpu["cuda"] and gpu["compute_cap"][0] >= 10:
        attn_impl = "sdpa"

    if attn_impl:
        try:
            config._attn_implementation = attn_impl
        except Exception:
            pass

    # 初始化模型（随机权重）
    _safe_update(task, 35, f"初始化 {arch.param_count_str()} 模型（随机权重）...")
    dtype = torch.bfloat16 if gpu["bf16"] else (torch.float16 if gpu["cuda"] else torch.float32)

    try:
        model = LlamaForCausalLM(config)
    except Exception as e:
        if "does not support" in str(e) and "attention" in str(e).lower():
            # SDPA 不支持 → 去掉 attn_implementation
            log(f"LlamaForCausalLM SDPA 不支持: {e}，使用默认 attention")
            try:
                config._attn_implementation = "eager"
                model = LlamaForCausalLM(config)
            except Exception:
                config._attn_implementation = None
                model = LlamaForCausalLM(config)
        else:
            raise

    # 转换精度
    model = model.to(dtype)
    if gpu["cuda"]:
        if arch.cpu_offload:
            # 大模型: gradient_checkpointing 必开, 不急着移到 GPU
            # Trainer 会通过 device_map 处理
            _safe_update(task, 37, f"⚡ CPU offload 模式（优化器状态卸载到内存）")
            model = model.to("cuda:0")
        else:
            model = model.to("cuda:0")

    # 大模型强制 gradient checkpointing
    if arch.cpu_offload or arch.param_count() > 200_000_000:
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                model.gradient_checkpointing_enable()
            log("✅ Gradient checkpointing 已启用（省显存）")

    total_params = sum(p.numel() for p in model.parameters())
    _offload_str = " | CPU offload" if arch.cpu_offload else ""
    log(f"模型构建完成: {total_params:,} 参数 ({total_params / 1e6:.1f}M){_offload_str}")
    _safe_update(task, 40, f"✅ 模型就绪: {total_params / 1e6:.1f}M 参数{_offload_str}")

    # 保存初始检查点
    if output_dir is None:
        output_dir = str(LORAS_DIR / f"pretrain_{arch.preset}_{int(time.time())}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    return model, tok, output_dir


# ════════════════════════════════════════════════════
#  预训练核心
# ════════════════════════════════════════════════════

def _prepare_corpus(
    corpus_paths: List[Path],
    tokenizer,
    max_seq_len: int,
    task=None,
):
    """将语料文件 tokenize 为固定长度序列。

    策略: 把所有文本拼成一个长序列 → 按 max_seq_len 切块。
    使用分批 tokenize 避免大语料 OOM。
    """
    from datasets import Dataset

    _safe_update(task, 45, "Tokenize 语料...")

    # ── 分批流式 tokenize（避免 all_tokens 列表 OOM）──
    # 先统计总 token 数，再分批构建 chunks
    token_buffer = []          # 当前未满的 buffer
    input_ids_all = []         # 最终 chunks
    attention_all = []
    total_tokens = 0
    pad_id = tokenizer.pad_token_id or 0
    eos_id = tokenizer.eos_token_id or 2

    def _flush_buffer():
        """把 buffer 中的 tokens 切成 max_seq_len 块"""
        nonlocal token_buffer
        while len(token_buffer) >= max_seq_len:
            chunk = token_buffer[:max_seq_len]
            input_ids_all.append(chunk)
            attention_all.append([1] * max_seq_len)
            token_buffer = token_buffer[max_seq_len:]

    def _feed_text(text: str):
        nonlocal total_tokens
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            return
        ids.append(eos_id)
        total_tokens += len(ids)
        token_buffer.extend(ids)
        # 每积累 10 万 tokens 就切一次，控制内存
        if len(token_buffer) >= 100_000:
            _flush_buffer()

    for p in corpus_paths:
        p = Path(p)
        try:
            if p.suffix in (".jsonl",):
                # 逐行流式读取，不把整个文件读入内存
                with open(p, "r", encoding="utf-8-sig") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            text = _extract_text(obj)
                            if text:
                                _feed_text(text)
                        except Exception:
                            continue
            elif p.suffix == ".json":
                raw = p.read_text(encoding="utf-8-sig")
                data = json.loads(raw)
                if isinstance(data, list):
                    for obj in data:
                        text = _extract_text(obj)
                        if text:
                            _feed_text(text)
            else:
                # txt, md 等纯文本：分块读取大文件
                _chunk_size = 1024 * 1024  # 1MB chunks
                with open(p, "r", encoding="utf-8-sig") as fh:
                    while True:
                        chunk = fh.read(_chunk_size)
                        if not chunk:
                            break
                        _feed_text(chunk)
        except Exception as e:
            log(f"语料文件读取失败 {p.name}: {e}")
            continue

    # 处理 buffer 中剩余的 tokens
    _flush_buffer()

    if total_tokens == 0:
        raise RuntimeError("语料 tokenize 后为空，请检查文件内容")

    log(f"语料总 tokens: {total_tokens:,} ({total_tokens / 1e6:.1f}M)")
    _safe_update(task, 50, f"语料: {total_tokens:,} tokens → {len(input_ids_all)} 块 (seq_len={max_seq_len})")

    if not input_ids_all:
        # 语料太短 → 至少保留一块（pad 补齐）
        padded = token_buffer + [pad_id] * (max_seq_len - len(token_buffer))
        return Dataset.from_dict({
            "input_ids": [padded[:max_seq_len]],
            "attention_mask": [[1] * len(token_buffer) + [0] * (max_seq_len - len(token_buffer))],
        }), total_tokens

    # 不再手动创建 labels — DataCollatorForLanguageModeling 会自动处理
    ds = Dataset.from_dict({
        "input_ids": input_ids_all,
        "attention_mask": attention_all,
    })

    log(f"预训练数据: {len(ds)} 块 × {max_seq_len} tokens = {len(ds) * max_seq_len:,} tokens")
    _safe_update(task, 55, f"✅ {len(ds)} 块 × {max_seq_len} = {len(ds) * max_seq_len:,} tokens")
    return ds, total_tokens


class PretrainEngine:
    """从零预训练引擎"""

    def pretrain(
        self,
        arch_config: ArchConfig,
        corpus_paths: List[str],
        tokenizer_path: str,
        output_name: str = "my_model",
        params: Dict[str, Any] = None,
        task=None,
    ) -> str:
        """从零构建并预训练模型。

        流程: 构建架构 → 加载 tokenizer → tokenize 语料 → 全参数训练
        """
        params = params or {}
        corpus_paths = [Path(p) for p in corpus_paths]

        _safe_update(task, 1, "🔨 从零构建模型...")

        # 1. 构建模型
        out_dir = str(LORAS_DIR / output_name)
        model, tok, out_dir = build_model(arch_config, tokenizer_path, out_dir, task)

        # 2. 准备语料
        max_seq = int(params.get("max_seq_len", 512))
        ds, total_tokens = _prepare_corpus(corpus_paths, tok, max_seq, task)

        if len(ds) == 0:
            raise RuntimeError("语料过短，无法训练")

        # 3. 训练
        return self._run_training(
            model, tok, ds, out_dir, total_tokens,
            params, task,
            mode="pretrain",
            arch_config=arch_config,
        )

    def continual_pretrain(
        self,
        base_model: str,
        corpus_paths: List[str],
        output_name: str = "cpt_model",
        params: Dict[str, Any] = None,
        task=None,
    ) -> str:
        """继续预训练: 在现有模型上用新语料全参数训练。

        适用: 领域适应（如医学、法律、金融）
        """
        params = params or {}
        corpus_paths = [Path(p) for p in corpus_paths]

        _safe_update(task, 1, f"📚 加载基座模型: {base_model}")

        import torch
        try:
            from core.safe_loader import ensure_model_importable
            ensure_model_importable(base_model)
        except Exception:
            pass
        from transformers import AutoModelForCausalLM, AutoTokenizer

        gpu = _detect_gpu()

        # 加载 tokenizer
        tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        # 加载模型（全参数，不用 LoRA）
        _safe_update(task, 10, "加载模型（全参数模式）...")
        mk = dict(trust_remote_code=True)
        if gpu["cuda"]:
            mk["device_map"] = {"": 0}
            # Blackwell 兼容
            if gpu["compute_cap"][0] >= 10:
                mk["attn_implementation"] = "sdpa"
        dtype = torch.bfloat16 if gpu["bf16"] else (torch.float16 if gpu["cuda"] else torch.float32)
        mk["torch_dtype"] = dtype

        try:
            # 预导入模型类（防止 LazyAutoMapping 失效）
            try:
                        from core.safe_loader import ensure_model_importable
                        ensure_model_importable(base_model)
            except Exception:
                        pass
            model = AutoModelForCausalLM.from_pretrained(base_model, **mk)
        except Exception as e:
            if "does not support" in str(e) and "attention" in str(e).lower():
                mk["attn_implementation"] = "eager"
                try:
                    model = AutoModelForCausalLM.from_pretrained(base_model, **mk)
                except Exception:
                    mk.pop("attn_implementation", None)
                    model = AutoModelForCausalLM.from_pretrained(base_model, **mk)
            else:
                raise

        # 启用 gradient checkpointing（省显存）
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                model.gradient_checkpointing_enable()
        model.config.use_cache = False

        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _safe_update(task, 20,
            f"✅ 模型: {total_params / 1e6:.0f}M 参数 (全部可训练)")

        # 准备语料
        max_seq = int(params.get("max_seq_len", 512))
        ds, total_tokens = _prepare_corpus(corpus_paths, tok, max_seq, task)

        out_dir = str(LORAS_DIR / output_name)
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        return self._run_training(
            model, tok, ds, out_dir, total_tokens,
            params, task,
            mode="cpt",
            base_model_name=base_model,
        )

    def _run_training(
        self,
        model, tok, ds, out_dir: str, total_tokens: int,
        params: Dict[str, Any],
        task,
        mode: str = "pretrain",
        arch_config: ArchConfig = None,
        base_model_name: str = None,
    ) -> str:
        """执行训练循环（pretrain 和 CPT 共用）"""
        import torch
        from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

        gpu = _detect_gpu()

        # 训练参数
        lr = float(params.get("lr", 1e-4 if mode == "cpt" else 3e-4))
        batch_size = int(params.get("batch_size", 1))
        epochs = float(params.get("epochs", 1))
        ga_steps = int(params.get("gradient_accumulation_steps", 8))
        warmup_ratio = float(params.get("warmup_ratio", 0.05))
        max_steps = params.get("max_steps")

        # CPU offload 模式: 强制保守参数
        _cpu_offload = arch_config.cpu_offload if arch_config else False
        if _cpu_offload:
            batch_size = 1
            ga_steps = max(ga_steps, 16)
            _safe_update(task, 56, "⚡ CPU offload: batch=1, GA≥16, 8bit 优化器")

        # VRAM 安全检查
        if gpu["cuda"] and gpu["vram_mb"] > 0:
            avail = gpu["vram_mb"]
            model_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
            # 8bit optimizer ≈ 0.25x model, gradients ≈ 1x model, activations ≈ variable
            overhead = model_mem * 2.5 + 500
            if _cpu_offload:
                overhead = model_mem * 2.0 + 500  # 8bit optimizer paging 更激进
            if overhead > avail * 0.95:
                _safe_update(task, 55, f"⚠️ VRAM 紧张 ({overhead:.0f}MB / {avail}MB)，已自动优化")
                batch_size = 1
                ga_steps = max(ga_steps, 16)

        # 精度
        _fp16 = not gpu["bf16"] and gpu["cuda"]
        _bf16 = gpu["bf16"]

        # 优化器
        optim = "paged_adamw_8bit" if gpu["cuda"] else "adamw_torch"

        # 分割验证集
        eval_ds = None
        if len(ds) > 50:
            split = ds.train_test_split(test_size=0.05, seed=42)
            ds, eval_ds = split["train"], split["test"]

        # Data collator（标准 CLM，不 mask 任何 token）
        collator = DataCollatorForLanguageModeling(
            tokenizer=tok, mlm=False,
        )

        # 构建参数
        args_dict = dict(
            output_dir=out_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=ga_steps,
            learning_rate=lr,
            num_train_epochs=epochs,
            max_steps=int(max_steps) if max_steps else -1,
            lr_scheduler_type=params.get("lr_scheduler_type", "cosine"),
            warmup_ratio=warmup_ratio,
            weight_decay=float(params.get("weight_decay", 0.01)),
            max_grad_norm=1.0,
            logging_steps=int(params.get("logging_steps", 10)),
            save_steps=int(params.get("save_steps", 500)),
            save_total_limit=2,
            fp16=_fp16, bf16=_bf16,
            report_to=[],
            remove_unused_columns=False,
            dataloader_num_workers=0,
            dataloader_pin_memory=gpu["cuda"],
            optim=optim,
        )

        # Eval 相关
        if eval_ds is not None:
            try:
                from transformers import TrainingArguments as _TA
                import inspect
                _p = set(inspect.signature(_TA.__init__).parameters.keys())
                if "eval_strategy" in _p:
                    args_dict["eval_strategy"] = "steps"
                elif "evaluation_strategy" in _p:
                    args_dict["evaluation_strategy"] = "steps"
                args_dict["eval_steps"] = max(args_dict["logging_steps"] * 5, 50)
            except Exception:
                eval_ds = None

        training_args = TrainingArguments(**args_dict)

        # Callbacks
        cbs = self._build_callbacks(task)

        _mode_str = "从零预训练" if mode == "pretrain" else "继续预训练 (CPT)"
        _safe_update(task, 60,
            f"🚀 {_mode_str} | lr={lr} | batch={batch_size}×{ga_steps} | "
            f"{total_tokens:,} tokens")

        trainer = Trainer(
            model=model,
            tokenizer=tok,
            args=training_args,
            train_dataset=ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            callbacks=cbs,
        )

        # 训练
        trainer.train()

        # 保存
        _safe_update(task, 95, "保存模型...")
        trainer.save_model(out_dir)
        tok.save_pretrained(out_dir)

        # 保存元信息
        total_params = sum(p.numel() for p in model.parameters())
        meta = {
            "mode": mode,
            "total_params": total_params,
            "total_tokens": total_tokens,
            "dataset_chunks": len(ds),
            "lr": lr,
            "batch_size": batch_size,
            "gradient_accumulation_steps": ga_steps,
            "epochs": epochs,
            "version": "v3.0",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if arch_config:
            meta["architecture"] = asdict(arch_config)
        if base_model_name:
            meta["base_model"] = base_model_name

        # 提取最终 loss
        try:
            if hasattr(trainer, "state") and hasattr(trainer.state, "log_history"):
                for entry in reversed(trainer.state.log_history):
                    if "loss" in entry:
                        meta["final_loss"] = entry["loss"]
                        break
                    if "train_loss" in entry:
                        meta["final_loss"] = entry["train_loss"]
                        break
        except Exception:
            pass

        (Path(out_dir) / "forgex_pretrain_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )

        _safe_update(task, 100, f"✅ {_mode_str}完成: {out_dir}")

        # 释放
        _cleanup(trainer, model, tok, ds, eval_ds)
        return out_dir

    def _build_callbacks(self, task):
        cbs = []
        try:
            from transformers import TrainerCallback

            class _Progress(TrainerCallback):
                def __init__(self, t):
                    self._t = t

                def on_log(self, args, state, control, logs=None, **kw):
                    if not self._t or not logs:
                        return
                    if state.max_steps > 0:
                        pct = min(60 + 35 * state.global_step / state.max_steps, 95)
                        loss = logs.get("loss", logs.get("train_loss", "?"))
                        _safe_update(self._t, pct,
                            f"Step {state.global_step}/{state.max_steps} | loss={loss}")
                    # Metrics for visualization
                    rec = {"step": int(state.global_step)}
                    for k, v in logs.items():
                        if isinstance(v, (int, float)):
                            rec[k] = float(v)
                    if len(rec) > 1:
                        try:
                            self._t.logs.append("[METRIC]" + json.dumps(rec, default=str))
                        except Exception:
                            pass

            cbs.append(_Progress(task))
        except Exception:
            pass
        return cbs


def _cleanup(*objects):
    for obj in objects:
        try:
            if hasattr(obj, "model"):
                obj.model = None
            if hasattr(obj, "optimizer"):
                obj.optimizer = None
            if hasattr(obj, "cpu"):
                obj.cpu()
        except Exception:
            pass
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ════════════════════════════════════════════════════
#  知识注入管线 — API 辅助知识蒸馏
# ════════════════════════════════════════════════════

class KnowledgeForge:
    """文档 → API 生成 QA → SFT 训练模型

    流程:
      1. 从知识文档提取文本并切块
      2. 每块送 API 生成多组高质量 QA 对（含 CoT 推理链）
      3. 汇总为 SFT 数据集
      4. 用该数据集 SFT 训练目标模型（LoRA 或全参数）

    支持: 任何 OpenAI 兼容 API（GPT / DeepSeek / Claude 代理 / 本地 Ollama）
    """

    def run(
        self,
        model_path: str,
        doc_paths: List[str],
        api_base: str,
        api_key: str,
        api_model: str,
        output_name: str = "knowledge_model",
        qa_per_chunk: int = 3,
        max_chunks: int = 200,
        train_method: str = "sft",
        train_params: Dict[str, Any] = None,
        system_prompt: str = "",
        use_cot: bool = True,
        task=None,
    ) -> str:
        """执行知识注入全流程。

        Args:
            model_path: 目标模型路径（从零构建的或 HF 模型）
            doc_paths: 知识文档路径列表
            api_base: API 基础 URL
            api_key: API 密钥
            api_model: API 模型名
            qa_per_chunk: 每块生成几个 QA
            max_chunks: 最多处理多少块
            use_cot: 是否要求 API 生成思维链
        """
        train_params = train_params or {}

        # 1. 提取文档文本
        _safe_update(task, 2, "📄 提取知识文档...")
        all_text = self._extract_docs(doc_paths)
        if not all_text.strip():
            raise RuntimeError("无法从文档中提取有效内容")

        # 2. 切块
        chunks = self._split_text(all_text)
        if len(chunks) > max_chunks:
            chunks = chunks[:max_chunks]
        _safe_update(task, 5, f"📚 {len(chunks)} 个知识块待处理 (总 {len(all_text):,} 字符)")

        # 3. API 生成 QA 数据
        qa_data = self._generate_qa_via_api(
            chunks, api_base, api_key, api_model,
            qa_per_chunk=qa_per_chunk,
            system_prompt=system_prompt,
            use_cot=use_cot,
            task=task,
        )
        if not qa_data:
            raise RuntimeError("API 未能生成有效的 QA 数据")

        # 4. 保存数据集
        ts = time.strftime("%Y%m%d_%H%M%S")
        ds_path = DATASETS_DIR / f"knowledge_{output_name}_{ts}.jsonl"
        with open(ds_path, "w", encoding="utf-8") as f:
            for item in qa_data:
                f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        log(f"✅ 知识 QA 数据: {len(qa_data)} 条 → {ds_path.name}")
        _safe_update(task, 60, f"✅ {len(qa_data)} 条 QA 数据已生成")

        # 5. SFT 训练
        _safe_update(task, 62, f"🔥 SFT 训练: {output_name}...")
        from core.trainer import TrainingEngine
        trainer = TrainingEngine()

        sft_params = {
            "output_name": output_name,
            "lr": float(train_params.get("lr", 2e-4)),
            "batch_size": int(train_params.get("batch_size", 2)),
            "epochs": float(train_params.get("epochs", 3)),
            "max_seq_len": int(train_params.get("max_seq_len", 2048)),
            "use_qlora": bool(train_params.get("use_qlora", False)),
            "rank": int(train_params.get("rank", 64)),
            "gradient_accumulation_steps": int(train_params.get("ga_steps", 4)),
            "use_dora": True,
            "use_rslora": True,
            "auto_clean": True,
            "label_smoothing": 0.1,
            "neftune_noise_alpha": 5.0,
        }

        result = trainer.train(
            method=train_method,
            backend="trl",
            base_model=model_path,
            dataset_path=[str(ds_path)],
            params=sft_params,
            task=task,
        )

        _safe_update(task, 100, f"✅ 知识注入完成: {output_name}")
        return result

    def generate_data_only(
        self,
        doc_paths: List[str],
        api_base: str,
        api_key: str,
        api_model: str,
        output_name: str = "knowledge_qa",
        qa_per_chunk: int = 3,
        max_chunks: int = 200,
        system_prompt: str = "",
        use_cot: bool = True,
        task=None,
    ) -> str:
        """仅生成 QA 数据，不训练。用于手动检查数据质量。"""

        _safe_update(task, 2, "📄 提取知识文档...")
        all_text = self._extract_docs(doc_paths)
        if not all_text.strip():
            raise RuntimeError("无法从文档中提取有效内容")

        chunks = self._split_text(all_text)
        if len(chunks) > max_chunks:
            chunks = chunks[:max_chunks]
        _safe_update(task, 5, f"📚 {len(chunks)} 块")

        qa_data = self._generate_qa_via_api(
            chunks, api_base, api_key, api_model,
            qa_per_chunk=qa_per_chunk,
            system_prompt=system_prompt,
            use_cot=use_cot,
            task=task,
        )
        if not qa_data:
            raise RuntimeError("API 未能生成有效 QA")

        ts = time.strftime("%Y%m%d_%H%M%S")
        ds_path = DATASETS_DIR / f"knowledge_{output_name}_{ts}.jsonl"
        with open(ds_path, "w", encoding="utf-8") as f:
            for item in qa_data:
                f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

        _safe_update(task, 100, f"✅ 数据已保存: {ds_path.name} ({len(qa_data)} 条)")
        return str(ds_path)

    # ──────────── 内部方法 ────────────

    def _extract_docs(self, doc_paths: List[str]) -> str:
        """从文件列表提取所有文本"""
        texts = []
        for dp in doc_paths:
            p = Path(dp)
            if not p.exists():
                continue
            try:
                if p.suffix.lower() in (".txt", ".md"):
                    texts.append(p.read_text(encoding="utf-8-sig"))
                elif p.suffix.lower() in (".jsonl", ".json"):
                    raw = p.read_text(encoding="utf-8-sig")
                    if p.suffix == ".jsonl":
                        for line in raw.strip().splitlines():
                            try:
                                obj = json.loads(line)
                                t = _extract_text(obj)
                                if t:
                                    texts.append(t)
                            except Exception:
                                continue
                    else:
                        data = json.loads(raw)
                        if isinstance(data, list):
                            for obj in data:
                                t = _extract_text(obj)
                                if t:
                                    texts.append(t)
                        elif isinstance(data, dict):
                            t = _extract_text(data)
                            if t:
                                texts.append(t)
                elif p.suffix.lower() == ".csv":
                    # CSV: 每行拼成文本
                    import csv
                    with open(p, "r", encoding="utf-8-sig") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            texts.append(" | ".join(f"{k}: {v}" for k, v in row.items() if v))
                else:
                    # 尝试当纯文本
                    content = p.read_text(encoding="utf-8-sig", errors="replace")
                    if len(content.strip()) > 50:
                        texts.append(content)
            except Exception as e:
                log(f"文档读取失败 {p.name}: {e}")
        return "\n\n".join(texts)

    def _split_text(self, text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
        """智能切块"""
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            # 尝试在句号/换行处断开
            if end < len(text):
                for sep in ["\n\n", "\n", "。", ". ", "！", "？"]:
                    pos = text.rfind(sep, start + chunk_size // 2, end + 200)
                    if pos > start:
                        end = pos + len(sep)
                        break
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap
        return chunks

    def _generate_qa_via_api(
        self,
        chunks: List[str],
        api_base: str,
        api_key: str,
        api_model: str,
        qa_per_chunk: int = 3,
        system_prompt: str = "",
        use_cot: bool = True,
        task=None,
    ) -> List[Dict]:
        """并行调用 API 从文本块生成 QA 对"""
        import concurrent.futures
        import threading

        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("需要安装 openai: pip install openai")

        client = OpenAI(base_url=api_base.rstrip("/"), api_key=api_key or "sk-placeholder")

        all_qa = []
        lock = threading.Lock()
        total = len(chunks)

        _cot_instruction = ""
        if use_cot:
            _cot_instruction = (
                "\n- 回答必须包含详细的推理过程（思维链），先分析再给结论"
                "\n- 如果涉及步骤，请逐步说明"
            )

        _sys = system_prompt.strip() if system_prompt else "你是一个知识提取专家，擅长从文本中生成高质量的问答训练数据。"

        def _process_chunk(idx: int, chunk: str) -> List[Dict]:
            prompt = (
                f"请根据以下知识文本，生成 {qa_per_chunk} 个高质量问答对。\n\n"
                f"要求:\n"
                f"- 问题要具体、有针对性，覆盖文本中的关键知识点\n"
                f"- 回答要准确、完整，忠于原文信息{_cot_instruction}\n"
                f"- 避免生成过于简单的是/否问题\n"
                f"- 每个 QA 用 JSON 格式，字段为 instruction 和 output\n\n"
                f"知识文本:\n{chunk[:2000]}\n\n"
                f"请严格用以下 JSON 数组格式回复（不要加 markdown 代码块）:\n"
                f'[{{"instruction": "问题1", "output": "回答1"}}, '
                f'{{"instruction": "问题2", "output": "回答2"}}]'
            )

            # 最多重试 3 次，指数退避
            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model=api_model,
                        messages=[
                            {"role": "system", "content": _sys},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.7,
                        max_tokens=4096,
                    )
                    text = resp.choices[0].message.content.strip()
                    result = self._parse_qa_response(text)
                    if result:
                        return result
                    # 解析失败但没异常 → 重试
                    if attempt < 2:
                        time.sleep(1)
                        continue
                    return []
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2 ** attempt)  # 1s, 2s
                        continue
                    log(f"API 调用失败 (chunk {idx}, 已重试3次): {e}")
                    return []

        # 并行调用
        workers = min(5, total)
        _safe_update(task, 10, f"🤖 API 生成 QA ({total} 块, {workers} 并行)...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as exe:
            futures = {}
            for i, chunk in enumerate(chunks):
                futures[exe.submit(_process_chunk, i, chunk)] = i

            done = 0
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    qa_items = future.result()
                    with lock:
                        all_qa.extend(qa_items)
                        done += 1
                    pct = 10 + 45 * done / total
                    _safe_update(task, pct,
                        f"🤖 API 进度: {done}/{total} 块 | 已生成 {len(all_qa)} 条 QA")
                except Exception as e:
                    done += 1
                    log(f"QA 生成异常 (chunk {idx}): {e}")

        log(f"API QA 生成完成: {len(all_qa)} 条")
        return all_qa

    def _parse_qa_response(self, text: str) -> List[Dict]:
        """解析 API 返回的 QA JSON"""
        # 去掉 markdown 代码块
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        results = []
        try:
            data = json.loads(text)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        inst = item.get("instruction") or item.get("question") or item.get("prompt", "")
                        out = item.get("output") or item.get("answer") or item.get("response", "")
                        if inst and out and len(inst) > 5 and len(out) > 10:
                            results.append({"instruction": inst.strip(), "output": out.strip()})
            elif isinstance(data, dict):
                inst = data.get("instruction") or data.get("question", "")
                out = data.get("output") or data.get("answer", "")
                if inst and out:
                    results.append({"instruction": inst.strip(), "output": out.strip()})
        except json.JSONDecodeError:
            # 尝试逐行解析
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("{"):
                    try:
                        obj = json.loads(line)
                        inst = obj.get("instruction") or obj.get("question", "")
                        out = obj.get("output") or obj.get("answer", "")
                        if inst and out:
                            results.append({"instruction": inst.strip(), "output": out.strip()})
                    except Exception:
                        pass
        return results


knowledge_forge = KnowledgeForge()
pretrain_engine = PretrainEngine()
