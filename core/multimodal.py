# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

"""ForgeX 多模态训练引擎 — 视觉 + 音频融合

为 8GB VRAM 设计的 LLaVA-style 多模态训练方案:
  1. 冻结视觉/音频编码器 (SigLIP / CLIP / Whisper)
  2. 训练投影层 (MLP bridge, ~10-30M 参数)
  3. LoRA 微调 LLM (Qwen / LLaMA 等)
  → 总显存: 编码器(frozen,推理) + LLM(LoRA) + 投影层 ≈ 4-6GB

数据格式:
  图文对: {"image": "xxx.jpg", "conversations": [...]}
  音文对: {"audio": "xxx.wav", "conversations": [...]}
  纯文本: {"conversations": [...]}  (兼容混合训练)
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core import DATASETS_DIR, LORAS_DIR, MODELS_CACHE_DIR, log


# ════════════════════════════════════════════════════
#  编码器配置
# ════════════════════════════════════════════════════

VISION_ENCODERS = {
    "siglip-base": {
        "label": "SigLIP-base (384dim, 快, 推荐)",
        "model_id": "google/siglip-base-patch16-224",
        "hidden_size": 768,
        "image_size": 224,
        "vram_frozen_mb": 350,
    },
    "clip-vit-b": {
        "label": "CLIP ViT-B/16 (512dim, 经典)",
        "model_id": "openai/clip-vit-base-patch16",
        "hidden_size": 768,
        "image_size": 224,
        "vram_frozen_mb": 350,
    },
    "clip-vit-l": {
        "label": "CLIP ViT-L/14 (768dim, 更强但更大)",
        "model_id": "openai/clip-vit-large-patch14",
        "hidden_size": 1024,
        "image_size": 224,
        "vram_frozen_mb": 900,
    },
    "siglip-so400m": {
        "label": "SigLIP-SO400M (1152dim, 最强视觉)",
        "model_id": "google/siglip-so400m-patch14-384",
        "hidden_size": 1152,
        "image_size": 384,
        "vram_frozen_mb": 1200,
    },
}

AUDIO_ENCODERS = {
    "whisper-tiny": {
        "label": "Whisper-tiny (384dim, 39M, 最快)",
        "model_id": "openai/whisper-tiny",
        "hidden_size": 384,
        "vram_frozen_mb": 150,
    },
    "whisper-base": {
        "label": "Whisper-base (512dim, 74M, 推荐)",
        "model_id": "openai/whisper-base",
        "hidden_size": 512,
        "vram_frozen_mb": 300,
    },
    "whisper-small": {
        "label": "Whisper-small (768dim, 244M)",
        "model_id": "openai/whisper-small",
        "hidden_size": 768,
        "vram_frozen_mb": 600,
    },
}

IMAGE_TOKEN = "<image>"
AUDIO_TOKEN = "<audio>"
TOOL_CALL_TOKEN = "<tool_call>"
TOOL_RESULT_TOKEN = "<tool_result>"


@dataclass
class MultimodalConfig:
    """多模态训练配置"""
    # 模态选择
    enable_vision: bool = True
    enable_audio: bool = False

    # 编码器
    vision_encoder: str = "siglip-base"
    audio_encoder: str = "whisper-base"

    # LLM
    base_model: str = "Qwen/Qwen2.5-0.5B"
    use_qlora: bool = False

    # 投影层
    projector_type: str = "mlp2x"  # "linear" | "mlp2x" | "mlp4x"
    projector_hidden: int = 0      # 0 = auto (llm_hidden * 2 for mlp)

    # 训练
    lr: float = 1e-3               # 投影层学习率
    lr_llm: float = 2e-5           # LLM LoRA 学习率
    batch_size: int = 1
    epochs: float = 3
    max_seq_len: int = 2048
    gradient_accumulation_steps: int = 8
    rank: int = 32                 # LoRA rank
    warmup_ratio: float = 0.03

    # 两阶段训练
    two_stage: bool = True         # True: 先冻结LLM训投影层 → 再一起训
    stage1_epochs: float = 1       # 阶段一: 只训投影层
    stage2_epochs: float = 2       # 阶段二: 投影层 + LoRA

    # ── AI 注意力窗口 ──
    # 控制模型在处理多模态输入时的注意力范围
    attention_window: int = 0      # 0 = 全局注意力, >0 = 滑动窗口 token 数
    cross_attn_every_n: int = 4    # 每 N 层插入一次跨模态注意力（0=用投影层代替）
    visual_token_budget: int = 256 # 每张图片最多占用的 token 数（影响注意力开销）

    # ── 工具链集成 ──
    enable_tool_use: bool = False  # 启用 function calling 训练
    tool_definitions: str = ""     # JSON 格式的工具定义（函数签名）
    tool_format: str = "chatml"    # 工具调用格式: "chatml" | "react" | "json"

    # 输出
    output_name: str = "multimodal_model"

    def estimate_vram_mb(self) -> int:
        """估算总 VRAM"""
        total = 500  # CUDA overhead

        if self.enable_vision:
            ve = VISION_ENCODERS.get(self.vision_encoder, {})
            total += ve.get("vram_frozen_mb", 400)

        if self.enable_audio:
            ae = AUDIO_ENCODERS.get(self.audio_encoder, {})
            total += ae.get("vram_frozen_mb", 300)

        # LLM (LoRA/QLoRA)
        # 粗估: 0.5B ≈ 1GB, 1.5B ≈ 3GB (with LoRA)
        total += 2000  # 典型小模型 + LoRA 开销

        # 投影层 (tiny)
        total += 50

        return total


def _safe_update(task, p: float, msg: str):
    if task is not None:
        try:
            task.update_progress(float(p), str(msg))
        except Exception:
            pass


# ════════════════════════════════════════════════════
#  投影层 (Projector)
# ════════════════════════════════════════════════════

def _build_projector(
    in_dim: int,
    out_dim: int,
    proj_type: str = "mlp2x",
    hidden_dim: int = 0,
):
    """构建投影层 MLP

    Args:
        in_dim: 编码器输出维度
        out_dim: LLM 隐藏维度
        proj_type: "linear" | "mlp2x" | "mlp4x"
    """
    import torch.nn as nn

    if proj_type == "linear":
        return nn.Linear(in_dim, out_dim)

    if hidden_dim <= 0:
        hidden_dim = out_dim * (2 if proj_type == "mlp2x" else 4)

    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )


# ════════════════════════════════════════════════════
#  多模态数据处理
# ════════════════════════════════════════════════════

class MultimodalDataset:
    """处理多模态训练数据

    支持格式:
      1. {"image": "path.jpg", "conversations": [{"role":"user","content":"<image>\n描述图片"},
                                                   {"role":"assistant","content":"这是..."}]}
      2. {"audio": "path.wav", "conversations": [{"role":"user","content":"<audio>\n转录上面的音频"},
                                                   {"role":"assistant","content":"..."}]}
      3. {"image": "path.jpg", "instruction": "描述这张图片", "output": "这是..."}
      4. 混合: 同时包含纯文本 + 图文 + 音文
    """

    @staticmethod
    def load(
        data_paths,
        base_dir: str = "",
    ) -> List[Dict]:
        """加载并验证多模态数据"""
        items = []
        base = Path(base_dir) if base_dir else None

        # 规范化输入
        if isinstance(data_paths, (str, Path)):
            data_paths = [str(data_paths)]

        for dp in data_paths:
            p = Path(dp)
            if not p.exists():
                log(f"⚠️ 数据文件不存在: {dp}")
                continue

            try:
                if p.suffix == ".jsonl":
                    with open(p, "r", encoding="utf-8-sig") as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                                item = MultimodalDataset._normalize(obj, base)
                                if item:
                                    items.append(item)
                            except Exception:
                                continue
                elif p.suffix == ".json":
                    data = json.loads(p.read_text(encoding="utf-8-sig"))
                    if isinstance(data, list):
                        for obj in data:
                            item = MultimodalDataset._normalize(obj, base)
                            if item:
                                items.append(item)
            except Exception as e:
                log(f"数据文件读取失败 {p.name}: {e}")

        log(f"多模态数据: {len(items)} 条")
        # 统计
        n_img = sum(1 for i in items if i.get("image"))
        n_aud = sum(1 for i in items if i.get("audio"))
        n_txt = sum(1 for i in items if not i.get("image") and not i.get("audio"))
        log(f"  图文: {n_img} | 音文: {n_aud} | 纯文本: {n_txt}")
        return items

    @staticmethod
    def _normalize(obj: Dict, base: Optional[Path]) -> Optional[Dict]:
        """标准化为统一格式"""
        result = {}

        # 图片路径
        img = obj.get("image") or obj.get("img") or obj.get("image_path")
        if img:
            img_path = Path(img)
            if base and not img_path.is_absolute():
                img_path = base / img_path
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"):
                result["image"] = str(img_path)

        # 音频路径
        aud = obj.get("audio") or obj.get("audio_path")
        if aud:
            aud_path = Path(aud)
            if base and not aud_path.is_absolute():
                aud_path = base / aud_path
            if aud_path.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
                result["audio"] = str(aud_path)

        # 对话格式
        convs = obj.get("conversations") or obj.get("messages")
        if convs and isinstance(convs, list):
            # 标准化 from/value → role/content
            normalized = []
            for m in convs:
                if isinstance(m, dict):
                    role = m.get("role") or m.get("from", "user")
                    content = m.get("content") or m.get("value", "")
                    msg_type = m.get("type", "")
                    entry = {"role": role, "content": content}
                    if msg_type:
                        entry["type"] = msg_type
                    normalized.append(entry)
            result["conversations"] = normalized
        elif obj.get("instruction"):
            # Alpaca 格式 → 转换
            user_msg = obj["instruction"]
            if result.get("image") and IMAGE_TOKEN not in user_msg:
                user_msg = IMAGE_TOKEN + "\n" + user_msg
            if result.get("audio") and AUDIO_TOKEN not in user_msg:
                user_msg = AUDIO_TOKEN + "\n" + user_msg
            if obj.get("input"):
                user_msg += "\n" + obj["input"]
            result["conversations"] = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": obj.get("output", "")},
            ]

        if not result.get("conversations"):
            return None
        # 至少有一个 assistant 回复
        assistant_roles = {"assistant", "gpt", "bot"}
        if not any(m.get("role") in assistant_roles for m in result["conversations"]):
            return None

        return result


# ════════════════════════════════════════════════════
#  多模态训练引擎
# ════════════════════════════════════════════════════

class MultimodalEngine:
    """LLaVA-style 多模态训练

    流程:
      1. 加载冻结的视觉/音频编码器
      2. 构建投影层 (encoder_dim → llm_dim)
      3. 加载 LLM + LoRA
      4. 阶段一: 冻结 LLM，只训投影层（学会对齐特征空间）
      5. 阶段二: 训投影层 + LoRA（学会理解+生成）
      6. 保存: LLM + LoRA + 投影层权重
    """

    def train(
        self,
        config: MultimodalConfig,
        data_paths,
        task=None,
    ) -> str:
        """执行多模态训练"""
        import torch
        import torch.nn as nn

        # 规范化数据路径 → 总是 List[str]
        if isinstance(data_paths, (str, Path)):
            data_paths = [str(data_paths)]
        elif not isinstance(data_paths, list):
            data_paths = [str(data_paths)]

        _safe_update(task, 1, "🔮 初始化多模态训练...")

        # ── 1. 检测 GPU ──
        gpu = self._detect_gpu()
        dtype = torch.bfloat16 if gpu["bf16"] else (torch.float16 if gpu["cuda"] else torch.float32)
        device = "cuda:0" if gpu["cuda"] else "cpu"

        # ── 2. 加载视觉编码器 ──
        vision_model = None
        vision_processor = None
        vision_dim = 0

        if config.enable_vision:
            _safe_update(task, 3, "👁️ 加载视觉编码器...")
            ve_info = VISION_ENCODERS.get(config.vision_encoder, VISION_ENCODERS["siglip-base"])
            vision_model, vision_processor, vision_dim = self._load_vision_encoder(
                ve_info["model_id"], dtype, device
            )
            log(f"视觉编码器: {config.vision_encoder} (dim={vision_dim})")

        # ── 3. 加载音频编码器 ──
        audio_model = None
        audio_processor = None
        audio_dim = 0

        if config.enable_audio:
            _safe_update(task, 8, "🔊 加载音频编码器...")
            ae_info = AUDIO_ENCODERS.get(config.audio_encoder, AUDIO_ENCODERS["whisper-base"])
            audio_model, audio_processor, audio_dim = self._load_audio_encoder(
                ae_info["model_id"], dtype, device
            )
            log(f"音频编码器: {config.audio_encoder} (dim={audio_dim})")

        # ── 4. 加载 LLM + Tokenizer ──
        _safe_update(task, 12, f"📝 加载 LLM: {config.base_model}...")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model, use_fast=True, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # 添加特殊 tokens
        new_tokens = []
        if config.enable_vision and IMAGE_TOKEN not in tokenizer.get_vocab():
            new_tokens.append(IMAGE_TOKEN)
        if config.enable_audio and AUDIO_TOKEN not in tokenizer.get_vocab():
            new_tokens.append(AUDIO_TOKEN)
        if config.enable_tool_use:
            for tt in (TOOL_CALL_TOKEN, TOOL_RESULT_TOKEN):
                if tt not in tokenizer.get_vocab():
                    new_tokens.append(tt)
        if new_tokens:
            tokenizer.add_tokens(new_tokens, special_tokens=True)
            log(f"添加特殊 tokens: {new_tokens}")

        # 加载模型
        mk = dict(trust_remote_code=True, torch_dtype=dtype)
        if gpu["cuda"]:
            mk["device_map"] = {"": 0}
        if config.use_qlora:
            try:
                from transformers import BitsAndBytesConfig
                mk["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=dtype,
                )
            except ImportError:
                log("⚠️ QLoRA 需要 bitsandbytes，回退到普通 LoRA")

        # Blackwell 兼容
        if gpu["cuda"] and gpu.get("compute_cap", (0,0))[0] >= 10:
            mk["attn_implementation"] = "sdpa"

        # ── 注意力窗口配置 ──
        if config.attention_window > 0:
            log(f"注意力窗口: {config.attention_window} tokens（滑动窗口注意力）")
            # 通过 config 注入 sliding window attention
            try:
                from transformers import AutoConfig
                model_cfg = AutoConfig.from_pretrained(config.base_model, trust_remote_code=True)
                if hasattr(model_cfg, "sliding_window"):
                    model_cfg.sliding_window = config.attention_window
                elif hasattr(model_cfg, "max_window_layers"):
                    # Qwen2 style
                    model_cfg.sliding_window = config.attention_window
                mk["config"] = model_cfg
            except Exception as e:
                log(f"⚠️ 滑动窗口设置失败（模型可能不支持）: {e}")

        try:
            llm = AutoModelForCausalLM.from_pretrained(config.base_model, **mk)
        except Exception as e:
            if "attention" in str(e).lower():
                mk.pop("attn_implementation", None)
                mk["attn_implementation"] = "eager"
                try:
                    llm = AutoModelForCausalLM.from_pretrained(config.base_model, **mk)
                except Exception:
                    mk.pop("attn_implementation", None)
                    llm = AutoModelForCausalLM.from_pretrained(config.base_model, **mk)
            else:
                raise

        # Resize embeddings for new tokens
        if new_tokens:
            llm.resize_token_embeddings(len(tokenizer))

        llm_dim = llm.config.hidden_size
        log(f"LLM: {config.base_model} (dim={llm_dim})")

        # ── 5. 构建投影层 ──
        _safe_update(task, 18, "🔗 构建投影层...")
        projectors = nn.ModuleDict()

        if config.enable_vision and vision_dim > 0:
            projectors["vision"] = _build_projector(
                vision_dim, llm_dim, config.projector_type, config.projector_hidden
            ).to(device).to(dtype)
            v_params = sum(p.numel() for p in projectors["vision"].parameters())
            log(f"视觉投影层: {v_params / 1e6:.1f}M 参数")

        if config.enable_audio and audio_dim > 0:
            projectors["audio"] = _build_projector(
                audio_dim, llm_dim, config.projector_type, config.projector_hidden
            ).to(device).to(dtype)
            a_params = sum(p.numel() for p in projectors["audio"].parameters())
            log(f"音频投影层: {a_params / 1e6:.1f}M 参数")

        # ── 6. 准备 LoRA ──
        _safe_update(task, 20, "🔧 配置 LoRA...")
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            r=config.rank,
            lora_alpha=config.rank * 2,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )

        # ── 7. 加载数据 ──
        _safe_update(task, 22, "📊 加载训练数据...")
        raw_data = MultimodalDataset.load(data_paths)
        if not raw_data:
            raise RuntimeError("没有有效的训练数据")

        # ── 8. 训练 ──
        out_dir = Path(LORAS_DIR) / config.output_name
        out_dir.mkdir(parents=True, exist_ok=True)

        if config.two_stage:
            # 阶段一: 只训投影层
            _safe_update(task, 25, "📐 阶段一: 训练投影层 (LLM 冻结)...")
            self._train_stage(
                llm=llm, tokenizer=tokenizer, projectors=projectors,
                vision_model=vision_model, vision_processor=vision_processor,
                audio_model=audio_model, audio_processor=audio_processor,
                data=raw_data, config=config,
                epochs=config.stage1_epochs,
                lr=config.lr, train_llm=False,
                lora_config=None,
                device=device, dtype=dtype, gpu=gpu,
                task=task, stage=1, out_dir=out_dir,
            )

            # 阶段二: 投影层 + LoRA
            _safe_update(task, 60, "🔥 阶段二: 投影层 + LoRA 联合训练...")
            self._train_stage(
                llm=llm, tokenizer=tokenizer, projectors=projectors,
                vision_model=vision_model, vision_processor=vision_processor,
                audio_model=audio_model, audio_processor=audio_processor,
                data=raw_data, config=config,
                epochs=config.stage2_epochs,
                lr=config.lr_llm, train_llm=True,
                lora_config=lora_config,
                device=device, dtype=dtype, gpu=gpu,
                task=task, stage=2, out_dir=out_dir,
            )
        else:
            # 单阶段: 投影层 + LoRA 一起训
            _safe_update(task, 25, "🔥 单阶段训练: 投影层 + LoRA...")
            self._train_stage(
                llm=llm, tokenizer=tokenizer, projectors=projectors,
                vision_model=vision_model, vision_processor=vision_processor,
                audio_model=audio_model, audio_processor=audio_processor,
                data=raw_data, config=config,
                epochs=config.epochs,
                lr=config.lr_llm, train_llm=True,
                lora_config=lora_config,
                device=device, dtype=dtype, gpu=gpu,
                task=task, stage=0, out_dir=out_dir,
            )

        # ── 9. 保存 ──
        _safe_update(task, 95, "💾 保存多模态模型...")
        self._save_model(
            llm, tokenizer, projectors, config, out_dir,
            vision_model, audio_model,
        )

        # 清理
        self._cleanup(llm, vision_model, audio_model, projectors)

        _safe_update(task, 100, f"✅ 多模态训练完成: {out_dir.name}")
        return str(out_dir)

    # ──────────── 训练阶段 ────────────

    def _train_stage(
        self,
        llm, tokenizer, projectors,
        vision_model, vision_processor,
        audio_model, audio_processor,
        data: List[Dict],
        config: MultimodalConfig,
        epochs: float,
        lr: float,
        train_llm: bool,
        lora_config,
        device: str,
        dtype,
        gpu: Dict,
        task,
        stage: int,
        out_dir: Path,
    ):
        """执行一个训练阶段"""
        import torch
        from torch.utils.data import DataLoader, Dataset as TorchDataset
        from torch.optim import AdamW

        # LoRA
        model = llm
        if train_llm and lora_config:
            from peft import get_peft_model
            if not hasattr(llm, "peft_config"):
                model = get_peft_model(llm, lora_config)
            else:
                model = llm
            model.print_trainable_parameters()
        else:
            # 冻结 LLM
            for p in llm.parameters():
                p.requires_grad = False

        # 投影层始终可训练
        for proj in projectors.values():
            for p in proj.parameters():
                p.requires_grad = True

        # 收集可训练参数
        train_params = list(p for proj in projectors.values() for p in proj.parameters() if p.requires_grad)
        if train_llm:
            train_params += [p for p in model.parameters() if p.requires_grad]

        total_trainable = sum(p.numel() for p in train_params)
        log(f"阶段{stage}: 可训练参数 {total_trainable / 1e6:.1f}M | LLM={'是' if train_llm else '否'}")

        # 优化器
        optimizer = AdamW(train_params, lr=lr, weight_decay=0.01)

        # 学习率 warmup 调度器
        from torch.optim.lr_scheduler import LambdaLR
        warmup_steps = max(int(total_steps * config.warmup_ratio), 1)
        def _lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(warmup_steps, 1))
            progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))  # cosine decay
        scheduler = LambdaLR(optimizer, _lr_lambda)

        # AMP for mixed precision
        use_amp = gpu.get("cuda", False) and dtype in (torch.float16, torch.bfloat16)
        scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16 and use_amp))

        # 简化训练循环（不用 HF Trainer，因为需要自定义 forward）
        model.train()
        ga_steps = config.gradient_accumulation_steps
        total_steps = int(len(data) * epochs / ga_steps)
        global_step = 0
        running_loss = 0.0

        # Gradient checkpointing
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                model.gradient_checkpointing_enable()
            model.config.use_cache = False

        base_pct = 25 if stage <= 1 else 60
        pct_range = 30 if stage <= 1 else 35

        for epoch in range(int(epochs) if epochs >= 1 else 1):
            epoch_steps = 0

            for i, item in enumerate(data):
                if epochs < 1 and i >= int(len(data) * epochs):
                    break

                try:
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype):
                        loss = self._forward_one(
                            item, model, tokenizer, projectors,
                            vision_model, vision_processor,
                            audio_model, audio_processor,
                            config, device, dtype,
                        )

                    if loss is None:
                        continue

                    loss = loss / ga_steps
                    scaler.scale(loss).backward()
                    running_loss += loss.item()
                    epoch_steps += 1

                    if epoch_steps % ga_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1

                        if global_step % 10 == 0:
                            avg_loss = running_loss / 10 if global_step >= 10 else running_loss / max(global_step, 1)
                            pct = base_pct + pct_range * global_step / max(total_steps, 1)
                            _safe_update(task, min(pct, base_pct + pct_range),
                                f"阶段{stage} | step {global_step}/{total_steps} | "
                                f"loss={avg_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")
                            running_loss = 0.0

                except Exception as e:
                    log(f"训练样本 {i} 失败: {e}")
                    optimizer.zero_grad()
                    continue

        # 最终 step
        if epoch_steps % ga_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        log(f"阶段{stage}完成: {global_step} steps")

        # 保存训练统计
        self._last_train_stats = getattr(self, "_last_train_stats", {})
        self._last_train_stats[f"stage{stage}_steps"] = global_step
        self._last_train_stats[f"stage{stage}_trainable_params"] = total_trainable
        if global_step > 0:
            self._last_train_stats["final_loss"] = running_loss / max(global_step % 10, 1)

    def _forward_one(
        self,
        item: Dict,
        model, tokenizer, projectors,
        vision_model, vision_processor,
        audio_model, audio_processor,
        config, device, dtype,
    ):
        """处理单个多模态样本的前向传播"""
        import torch

        conversations = item.get("conversations", [])
        if not conversations:
            return None

        # 构建文本输入
        text_parts = []
        for msg in conversations:
            role = msg.get("role", msg.get("from", "user"))
            content = msg.get("content", msg.get("value", ""))

            # ── 工具链集成: 处理 tool_call / tool_result 特殊消息 ──
            if config.enable_tool_use:
                if role == "tool_call" or msg.get("type") == "tool_call":
                    text_parts.append(
                        f"<|im_start|>assistant\n{TOOL_CALL_TOKEN}\n{content}\n<|im_end|>")
                    continue
                elif role == "tool_result" or msg.get("type") == "tool_result":
                    text_parts.append(
                        f"<|im_start|>tool\n{TOOL_RESULT_TOKEN}\n{content}\n<|im_end|>")
                    continue

            # 标准化 role
            if role in ("gpt", "assistant", "bot"):
                role = "assistant"
            elif role in ("human", "user"):
                role = "user"
            text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        # 如果启用工具链，在 system prompt 中注入工具定义
        if config.enable_tool_use and config.tool_definitions:
            tool_system = f"<|im_start|>system\n你可以使用以下工具:\n{config.tool_definitions}\n<|im_end|>"
            text_parts.insert(0, tool_system)

        full_text = "\n".join(text_parts)

        # Tokenize
        tokens = tokenizer(full_text, return_tensors="pt", truncation=True,
                           max_length=config.max_seq_len, padding=False)
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        # 获取 LLM embeddings
        if hasattr(model, "get_input_embeddings"):
            embed_layer = model.get_input_embeddings()
        elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            embed_layer = model.model.embed_tokens
        else:
            embed_layer = model.get_input_embeddings()

        inputs_embeds = embed_layer(input_ids)  # [1, seq, dim]

        # 处理视觉输入（带 token 预算）
        if item.get("image") and "vision" in projectors and vision_model is not None:
            try:
                img_embeds = self._encode_image(
                    item["image"], vision_model, vision_processor,
                    projectors["vision"], device, dtype,
                    token_budget=config.visual_token_budget,
                )
                if img_embeds is not None:
                    # 找到 <image> token 位置并替换
                    inputs_embeds, attention_mask = self._insert_embeds(
                        input_ids, inputs_embeds, attention_mask,
                        img_embeds, IMAGE_TOKEN, tokenizer,
                    )
            except Exception as e:
                log(f"图像处理失败: {e}")

        # 处理音频输入
        if item.get("audio") and "audio" in projectors and audio_model is not None:
            try:
                aud_embeds = self._encode_audio(
                    item["audio"], audio_model, audio_processor,
                    projectors["audio"], device, dtype,
                )
                if aud_embeds is not None:
                    inputs_embeds, attention_mask = self._insert_embeds(
                        input_ids, inputs_embeds, attention_mask,
                        aud_embeds, AUDIO_TOKEN, tokenizer,
                    )
            except Exception as e:
                log(f"音频处理失败: {e}")

        # 前向传播 — labels 必须与 inputs_embeds 同长
        labels = input_ids.clone()
        # Mask 非 assistant 部分
        labels = self._mask_non_assistant(labels, tokenizer)

        # 如果 modal embed 被插入，inputs_embeds 比 labels 长
        # 需要在插入位置用 -100（忽略 loss）填充 labels
        seq_diff = inputs_embeds.shape[1] - labels.shape[1]
        if seq_diff > 0:
            import torch
            pad_labels = torch.full(
                (labels.shape[0], seq_diff), -100,
                dtype=labels.dtype, device=labels.device
            )
            # Modal embeds 被插入在开头或 <image>/<audio> 位置
            # 保守做法: 把 -100 padding 放在前面
            labels = torch.cat([pad_labels, labels], dim=1)
        elif seq_diff < 0:
            labels = labels[:, :inputs_embeds.shape[1]]

        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask[:, :inputs_embeds.shape[1]],
            labels=labels,
        )
        return outputs.loss

    # ──────────── 编码器 ────────────

    def _load_vision_encoder(self, model_id: str, dtype, device):
        """加载视觉编码器（冻结）"""
        try:
            from transformers import SiglipVisionModel, SiglipImageProcessor
            model = SiglipVisionModel.from_pretrained(model_id, torch_dtype=dtype)
            processor = SiglipImageProcessor.from_pretrained(model_id)
        except Exception:
            from transformers import CLIPVisionModel, CLIPImageProcessor
            model = CLIPVisionModel.from_pretrained(model_id, torch_dtype=dtype)
            processor = CLIPImageProcessor.from_pretrained(model_id)

        model = model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        hidden_size = model.config.hidden_size
        return model, processor, hidden_size

    def _load_audio_encoder(self, model_id: str, dtype, device):
        """加载音频编码器（冻结）"""
        from transformers import WhisperModel, WhisperProcessor

        model = WhisperModel.from_pretrained(model_id, torch_dtype=dtype)
        processor = WhisperProcessor.from_pretrained(model_id)

        # 只用 encoder 部分
        encoder = model.encoder
        encoder = encoder.to(device)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False

        hidden_size = model.config.d_model
        return encoder, processor, hidden_size

    def _encode_image(self, image_path, vision_model, processor, projector, device, dtype,
                      token_budget: int = 0):
        """编码图片 → 投影到 LLM 空间

        Args:
            token_budget: 最大视觉 token 数。0=不限（保留所有 patch tokens）。
                         >0 时通过均匀采样裁剪 patch tokens，降低注意力开销。
        """
        import torch
        from PIL import Image

        if not Path(image_path).exists():
            log(f"⚠️ 图片不存在: {image_path}")
            return None

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            log(f"⚠️ 图片读取失败 {image_path}: {e}")
            return None

        inputs = processor(images=img, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = vision_model(**inputs)
            # 取最后一层隐藏状态（所有 patch tokens）
            if hasattr(outputs, "last_hidden_state"):
                features = outputs.last_hidden_state  # [1, num_patches, dim]
            else:
                features = outputs.pooler_output.unsqueeze(1)

        # 投影
        projected = projector(features.to(dtype))  # [1, num_patches, llm_dim]
        result = projected.squeeze(0)  # [num_patches, llm_dim]

        # ── 视觉 token 预算裁剪 ──
        if token_budget > 0 and result.shape[0] > token_budget:
            # 均匀采样，保留空间分布
            indices = torch.linspace(0, result.shape[0] - 1, token_budget).long()
            result = result[indices]

        return result

    def _encode_audio(self, audio_path, audio_encoder, processor, projector, device, dtype):
        """编码音频 → 投影到 LLM 空间"""
        import torch
        import numpy as np

        try:
            # 尝试 soundfile
            import soundfile as sf
            audio, sr = sf.read(audio_path)
        except ImportError:
            # 尝试 librosa
            try:
                import librosa
                audio, sr = librosa.load(audio_path, sr=16000)
            except ImportError:
                log("⚠️ 需要 soundfile 或 librosa 来处理音频")
                return None

        # Resample to 16kHz if needed
        if sr != 16000:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            except ImportError:
                log("⚠️ 需要 librosa 来重采样音频")
                return None

        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            audio = audio.mean(axis=1)

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device).to(dtype)

        with torch.no_grad():
            outputs = audio_encoder(input_features)
            features = outputs.last_hidden_state  # [1, time_steps, dim]

        projected = projector(features.to(dtype))
        return projected.squeeze(0)  # [time_steps, llm_dim]

    def _insert_embeds(self, input_ids, inputs_embeds, attention_mask,
                       modal_embeds, token_str, tokenizer):
        """在 <image> 或 <audio> token 位置插入编码器输出"""
        import torch

        token_id = tokenizer.convert_tokens_to_ids(token_str)
        if token_id is None or token_id == tokenizer.unk_token_id:
            # Token 未找到 → 直接拼接在开头
            inputs_embeds = torch.cat([
                modal_embeds.unsqueeze(0),
                inputs_embeds,
            ], dim=1)
            new_mask = torch.ones(1, modal_embeds.shape[0],
                                  device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([new_mask, attention_mask], dim=1)
            return inputs_embeds, attention_mask

        # 找到 token 位置
        positions = (input_ids[0] == token_id).nonzero(as_tuple=True)[0]
        if len(positions) == 0:
            # 没找到 → 拼接在开头
            inputs_embeds = torch.cat([
                modal_embeds.unsqueeze(0), inputs_embeds
            ], dim=1)
            new_mask = torch.ones(1, modal_embeds.shape[0],
                                  device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([new_mask, attention_mask], dim=1)
            return inputs_embeds, attention_mask

        # 替换第一个 token 位置
        pos = positions[0].item()
        seq_len = inputs_embeds.shape[1]
        n_modal = modal_embeds.shape[0]

        new_embeds = torch.cat([
            inputs_embeds[:, :pos],
            modal_embeds.unsqueeze(0),
            inputs_embeds[:, pos + 1:],
        ], dim=1)

        new_mask = torch.cat([
            attention_mask[:, :pos],
            torch.ones(1, n_modal, device=attention_mask.device, dtype=attention_mask.dtype),
            attention_mask[:, pos + 1:],
        ], dim=1)

        return new_embeds, new_mask

    def _mask_non_assistant(self, labels, tokenizer):
        """Mask 非 assistant 回复的 tokens（设为 -100）"""
        # 简化: 找 <|im_start|>assistant 和 <|im_end|> 之间的区域
        import torch

        label_ids = labels[0].tolist()
        text = tokenizer.decode(label_ids)

        # 标记 assistant 回复区域
        mask = torch.full_like(labels, -100)

        # 找所有 assistant 区域
        im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")

        if im_start is None or im_end is None:
            # 无法定位 → 全部参与 loss
            return labels

        in_assistant = False
        for i, tid in enumerate(label_ids):
            if tid == im_start:
                in_assistant = False
                # 检查后面是否跟着 "assistant"
                remaining = tokenizer.decode(label_ids[i:i + 5])
                if "assistant" in remaining.lower():
                    in_assistant = True
            elif tid == im_end:
                if in_assistant:
                    mask[0, i] = tid  # im_end 也参与 loss
                in_assistant = False
            elif in_assistant:
                mask[0, i] = tid

        return mask

    # ──────────── 保存/清理 ────────────

    def _save_model(self, llm, tokenizer, projectors, config, out_dir,
                    vision_model=None, audio_model=None):
        """保存多模态模型"""
        import torch

        # 保存 LLM + LoRA
        if hasattr(llm, "save_pretrained"):
            llm.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

        # 保存投影层
        proj_dir = out_dir / "projectors"
        proj_dir.mkdir(exist_ok=True)
        for name, proj in projectors.items():
            torch.save(proj.state_dict(), str(proj_dir / f"{name}_projector.pt"))

        # 保存元信息
        meta = {
            "type": "multimodal",
            "version": "v3.1",
            "base_model": config.base_model,
            "enable_vision": config.enable_vision,
            "enable_audio": config.enable_audio,
            "vision_encoder": config.vision_encoder if config.enable_vision else None,
            "audio_encoder": config.audio_encoder if config.enable_audio else None,
            "projector_type": config.projector_type,
            "lora_rank": config.rank,
            "two_stage": config.two_stage,
            "lr": config.lr,
            "lr_llm": config.lr_llm,
            "epochs": config.epochs,
            "attention_window": config.attention_window,
            "cross_attn_every_n": config.cross_attn_every_n,
            "visual_token_budget": config.visual_token_budget,
            "enable_tool_use": config.enable_tool_use,
            "tool_format": config.tool_format if config.enable_tool_use else None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        # 记录训练数据统计
        if hasattr(self, "_last_train_stats"):
            meta.update(self._last_train_stats)
        (out_dir / "forgex_multimodal_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        log(f"多模态模型已保存: {out_dir}")

    def _detect_gpu(self) -> Dict:
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

    def _cleanup(self, *objects):
        for obj in objects:
            try:
                if obj is not None:
                    del obj
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


multimodal_engine = MultimodalEngine()
