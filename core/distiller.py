# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

"""ForgeX 知识蒸馏引擎 — 以小博大的核心武器。

三大能力:
  1. Logit 蒸馏 — 学生学习教师的 soft output 分布 (KL Divergence)
  2. 多层特征蒸馏 — 学生中间层对齐教师中间层 (MSE / CosineEmbedding)
  3. 选择性神经元激活 — 稀疏化训练，让小模型专注核心能力

典型效果:
  - 0.5B 学生 + 7B 教师 → 学生可逼近 1.5B~3B 的效果
  - 1.5B 学生 + 14B 教师 → 学生可逼近 7B 的效果
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core import LORAS_DIR, DATASETS_DIR, log


# ================================================================
#  工具函数 — 显存监控 / 路径修复
# ================================================================

def _vram_status() -> Dict[str, Any]:
    """返回当前 GPU 显存状态 (MB)"""
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_mem / 1024**2
            alloc = torch.cuda.memory_allocated(0) / 1024**2
            reserved = torch.cuda.memory_reserved(0) / 1024**2
            free = total - reserved
            return {"total": total, "alloc": alloc, "reserved": reserved,
                    "free": free, "gpu": torch.cuda.get_device_name(0)}
    except Exception:
        pass
    return {"total": 0, "alloc": 0, "reserved": 0, "free": 0, "gpu": "N/A"}


def _vram_log(label: str = "") -> str:
    """打印并返回显存状态字符串"""
    v = _vram_status()
    if v["total"] == 0:
        return ""
    msg = f"[GPU] {v['gpu']} | {v['alloc']:.0f}MB / {v['total']:.0f}MB (剩余 {v['free']:.0f}MB)"
    if label:
        msg = f"{label} {msg}"
    log(msg)
    return msg


def _estimate_model_vram_mb(model_path: str) -> float:
    """粗估模型加载所需显存 (MB) — 根据 config.json 中的参数量估算"""
    try:
        cfg_file = Path(model_path) / "config.json"
        if not cfg_file.exists():
            return 0
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        # 用隐藏维度和层数粗估参数量
        hidden = cfg.get("hidden_size", 0)
        layers = cfg.get("num_hidden_layers", 0)
        vocab = cfg.get("vocab_size", 32000)
        intermediate = cfg.get("intermediate_size", hidden * 4)
        if hidden and layers:
            # 粗估: embedding + layers*(attention + FFN) ≈ params
            params_m = (vocab * hidden + layers * (4 * hidden * hidden + 3 * hidden * intermediate)) / 1e6
            return params_m * 2  # fp16 每参数 2 bytes → MB
    except Exception:
        pass
    return 0


def _repair_base_model_path(stale_path: str) -> Optional[str]:
    """尝试修复过时的 base_model_name_or_path

    场景: adapter_config.json 中记录的路径在当前机器上不存在
    策略:
      1. 路径已存在 → 直接用
      2. 提取模型ID (如 Qwen/Qwen2.5-1.5B-Instruct) → 搜索 HF cache
      3. 搜索 data/loras 下的同名目录
    """
    if not stale_path:
        return None

    # 1. 路径直接可用
    p = Path(stale_path)
    if p.is_dir() and (p / "config.json").exists():
        return stale_path

    # 2. 可能是 HF hub ID (如 "Qwen/Qwen2.5-1.5B-Instruct")
    if "/" in stale_path and not stale_path.startswith(("/", "\\", "C:", "D:")):
        # 看起来像 HF ID，让 transformers 去下载
        return stale_path

    # 3. 从过时路径中提取模型名，搜索 HF cache
    try:
        model_name = Path(stale_path).name
        # 搜索 HF cache
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        if hf_cache.exists():
            for d in hf_cache.iterdir():
                if d.is_dir() and model_name.lower() in d.name.lower():
                    snapshot_dir = d / "snapshots"
                    if snapshot_dir.exists():
                        snapshots = sorted(snapshot_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
                        for snap in snapshots:
                            if (snap / "config.json").exists():
                                log(f"  🔧 修复路径: {stale_path} → {snap}")
                                return str(snap)
        # 搜索 data/loras 下的合并目录
        loras_dir = Path(LORAS_DIR)
        if loras_dir.exists():
            for d in loras_dir.iterdir():
                if d.is_dir() and model_name.lower() in d.name.lower() and (d / "config.json").exists():
                    log(f"  🔧 修复路径: {stale_path} → {d}")
                    return str(d)
    except Exception:
        pass

    return None


# ================================================================
#  数据结构
# ================================================================

@dataclass
class DistillConfig:
    """蒸馏配置"""
    # 基本
    teacher_model: str = ""          # 教师模型 (HF ID 或本地路径)
    student_model: str = ""          # 学生模型
    dataset_path: str = ""           # 训练数据
    output_name: str = "distilled"   # 输出名

    # 蒸馏超参
    temperature: float = 4.0         # 蒸馏温度 (越高越 soft, 典型 2-8)
    alpha_ce: float = 0.5            # 交叉熵 loss 权重 (task loss)
    alpha_kd: float = 0.5            # KL divergence 权重 (distillation loss)
    alpha_feat: float = 0.0          # 特征蒸馏权重 (0=关闭, 0.1-0.3 为佳)

    # 多层蒸馏
    enable_layer_distill: bool = False
    layer_mapping: str = "auto"      # "auto" | "even" | "last_n" | "1:2,3:6,5:10"
    feature_loss_type: str = "mse"   # "mse" | "cosine"

    # 选择性神经元
    enable_sparse_activation: bool = False
    sparsity_target: float = 0.3     # 30% 稀疏度 (关闭30%神经元)
    sparse_warmup_ratio: float = 0.1 # 前10%步数逐步增加稀疏度

    # 训练参数
    lr: float = 2e-4
    batch_size: int = 1
    epochs: float = 3.0
    max_seq_len: int = 2048
    rank: int = 64                   # LoRA rank (学生用 LoRA 训练)
    gradient_accumulation_steps: int = 4
    use_qlora: bool = False

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


# ================================================================
#  蒸馏 Trainer
# ================================================================

class DistillationTrainer:
    """ForgeX 知识蒸馏训练器"""

    def train(self, config: DistillConfig, task=None) -> str:
        """执行知识蒸馏训练"""
        progress_cb = task.update_progress if task else lambda p, m="": log(m)

        progress_cb(2, "🧪 知识蒸馏 — 加载依赖...")

        # 延迟导入（避免启动慢）
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, AutoConfig,
            TrainingArguments, Trainer as HFTrainer,
        )
        from datasets import Dataset

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            if cap[0] >= 8:
                dtype = torch.bfloat16

        # ==============================================================
        # Step 1: 加载教师模型 (frozen, eval mode, 不用 LoRA)
        # ==============================================================

        # ═══ VRAM 预检查: 知识蒸馏需要同时加载教师+学生 ═══
        if torch.cuda.is_available():
            vram = _vram_status()
            est_teacher = _estimate_model_vram_mb(config.teacher_model)
            est_student = _estimate_model_vram_mb(config.student_model)
            est_total = est_teacher + est_student
            if est_total > 0 and est_total > vram["total"] * 0.85:
                log(f"⚠️ 显存预警: 教师({est_teacher:.0f}MB) + 学生({est_student:.0f}MB) "
                    f"= {est_total:.0f}MB > GPU {vram['total']:.0f}MB")
                log(f"  建议: 使用 API 蒸馏模式（无需本地加载教师模型）")

        progress_cb(5, f"📚 加载教师模型: {config.teacher_model}")
        # 预导入模型类（防止 LazyAutoMapping 失效）
        try:
                from core.safe_loader import ensure_model_importable
                ensure_model_importable(teacher)
        except Exception:
                pass
        teacher = AutoModelForCausalLM.from_pretrained(
            config.teacher_model,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        teacher_hidden = teacher.config.hidden_size
        teacher_layers = teacher.config.num_hidden_layers
        log(f"教师: hidden={teacher_hidden}, layers={teacher_layers}")

        # ==============================================================
        # Step 2: 加载学生模型 (+ LoRA)
        # ==============================================================
        progress_cb(15, f"🎓 加载学生模型: {config.student_model}")
        student = AutoModelForCausalLM.from_pretrained(
            config.student_model,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config.student_model, trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        student_hidden = student.config.hidden_size
        student_layers = student.config.num_hidden_layers
        log(f"学生: hidden={student_hidden}, layers={student_layers}")

        # 给学生加 LoRA
        progress_cb(20, "给学生加 LoRA...")
        try:
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(
                r=config.rank,
                lora_alpha=config.rank * 2,
                target_modules="all-linear",
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            student = get_peft_model(student, lora_cfg)
        except (ValueError, KeyError):
            from peft import LoraConfig, get_peft_model
            targets = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]
            lora_cfg = LoraConfig(
                r=config.rank,
                lora_alpha=config.rank * 2,
                target_modules=targets,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            student = get_peft_model(student, lora_cfg)

        trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
        total = sum(p.numel() for p in student.parameters())
        progress_cb(22, f"LoRA: trainable {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")

        # ==============================================================
        # Step 3: 多层蒸馏 — 构建投影层
        # ==============================================================
        projectors = None
        layer_map = []
        if config.enable_layer_distill and config.alpha_feat > 0:
            progress_cb(25, "🔗 构建层间映射...")
            layer_map = self._compute_layer_mapping(
                student_layers, teacher_layers, config.layer_mapping,
            )
            log(f"层映射: {layer_map}")

            # 如果维度不同，需要投影层
            if student_hidden != teacher_hidden:
                projectors = nn.ModuleList([
                    nn.Linear(student_hidden, teacher_hidden, bias=False).to(device).to(dtype)
                    for _ in layer_map
                ])
                log(f"投影层: {student_hidden} → {teacher_hidden} × {len(layer_map)}")

        # ==============================================================
        # Step 4: 稀疏激活掩码
        # ==============================================================
        sparse_masks = {}
        if config.enable_sparse_activation and config.sparsity_target > 0:
            progress_cb(27, f"🎯 选择性神经元: 稀疏度 {config.sparsity_target:.0%}")
            sparse_masks = self._init_sparse_masks(
                student, config.sparsity_target, device, dtype,
            )
            log(f"稀疏掩码: {len(sparse_masks)} 层")

        # ==============================================================
        # Step 5: 加载数据集
        # ==============================================================
        progress_cb(30, "📋 加载训练数据...")
        ds_path = Path(config.dataset_path)
        if not ds_path.is_absolute():
            ds_path = Path(DATASETS_DIR) / config.dataset_path
        if not ds_path.exists():
            raise FileNotFoundError(f"数据集不存在: {ds_path}")

        dataset = self._load_dataset(ds_path, tokenizer, config.max_seq_len)
        progress_cb(35, f"数据集: {len(dataset)} 条样本")

        # ==============================================================
        # Step 6: 自定义蒸馏 Training Loop
        # ==============================================================
        out_dir = LORAS_DIR / config.output_name
        out_dir.mkdir(parents=True, exist_ok=True)

        progress_cb(38, "⚙️ 配置训练...")

        # 优化器
        optim_params = list(student.parameters())
        if projectors is not None:
            optim_params += list(projectors.parameters())

        optimizer = self._build_optimizer(optim_params, config.lr, device)

        total_steps = int(
            math.ceil(len(dataset) / max(config.batch_size, 1))
            * config.epochs
            / max(config.gradient_accumulation_steps, 1)
        )
        scheduler = self._build_scheduler(optimizer, total_steps)

        progress_cb(40, f"🚀 开始蒸馏训练 ({total_steps} 步)...")

        # Training loop
        student.train()
        global_step = 0
        accum_loss = 0.0
        best_loss = float("inf")

        T = config.temperature
        alpha_ce = config.alpha_ce
        alpha_kd = config.alpha_kd
        alpha_feat = config.alpha_feat if config.enable_layer_distill else 0.0

        # DataLoader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True,
            num_workers=0, pin_memory=False,
        )

        for epoch in range(int(math.ceil(config.epochs))):
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # ---- 教师前向 (no grad) ----
                with torch.no_grad():
                    teacher_out = teacher(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=bool(layer_map),
                    )
                    teacher_logits = teacher_out.logits

                # ---- 学生前向 ----
                student_out = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_hidden_states=bool(layer_map),
                )
                student_logits = student_out.logits
                ce_loss = student_out.loss  # 标准交叉熵

                # ---- KL Divergence (Logit 蒸馏) ----
                # 对 vocab 维度做 softmax，温度缩放
                student_probs = F.log_softmax(student_logits / T, dim=-1)
                teacher_probs = F.softmax(teacher_logits / T, dim=-1)

                # 截取到相同 vocab size（学生和教师 vocab 可能不同）
                min_vocab = min(student_probs.size(-1), teacher_probs.size(-1))
                kd_loss = F.kl_div(
                    student_probs[..., :min_vocab],
                    teacher_probs[..., :min_vocab],
                    reduction="batchmean",
                ) * (T * T)

                # ---- 多层特征蒸馏 ----
                feat_loss = torch.tensor(0.0, device=device)
                if layer_map and alpha_feat > 0:
                    s_hidden = student_out.hidden_states
                    t_hidden = teacher_out.hidden_states
                    for i, (s_idx, t_idx) in enumerate(layer_map):
                        s_h = s_hidden[s_idx]  # (B, seq, student_hidden)
                        t_h = t_hidden[t_idx]  # (B, seq, teacher_hidden)
                        if projectors is not None:
                            s_h = projectors[i](s_h)
                        # 截取到相同 seq_len
                        min_len = min(s_h.size(1), t_h.size(1))
                        s_h = s_h[:, :min_len]
                        t_h = t_h[:, :min_len]
                        if config.feature_loss_type == "cosine":
                            feat_loss = feat_loss + (1 - F.cosine_similarity(
                                s_h.reshape(-1, s_h.size(-1)),
                                t_h.reshape(-1, t_h.size(-1)),
                            ).mean())
                        else:
                            feat_loss = feat_loss + F.mse_loss(s_h.float(), t_h.float())
                    feat_loss = feat_loss / max(len(layer_map), 1)

                # ---- 稀疏激活 ----
                if sparse_masks and config.enable_sparse_activation:
                    # 逐步增加稀疏度（warmup）
                    progress_ratio = global_step / max(total_steps, 1)
                    warmup = config.sparse_warmup_ratio
                    if progress_ratio < warmup:
                        current_sparsity = config.sparsity_target * (progress_ratio / warmup)
                    else:
                        current_sparsity = config.sparsity_target
                    self._apply_sparse_masks(student, sparse_masks, current_sparsity)

                # ---- 总 Loss ----
                loss = alpha_ce * ce_loss + alpha_kd * kd_loss + alpha_feat * feat_loss
                loss = loss / config.gradient_accumulation_steps
                loss.backward()

                accum_loss += loss.item()

                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in student.parameters() if p.requires_grad], 1.0,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % max(total_steps // 20, 1) == 0 or global_step == total_steps:
                        pct = min(40 + int(55 * global_step / max(total_steps, 1)), 95)
                        avg_loss = accum_loss / max(global_step, 1)
                        progress_cb(pct,
                            f"Step {global_step}/{total_steps} | "
                            f"loss={avg_loss:.4f} "
                            f"(CE={ce_loss.item():.3f} KD={kd_loss.item():.3f}"
                            f"{f' Feat={feat_loss.item():.3f}' if alpha_feat > 0 else ''})"
                        )

                    # 保存最佳
                    if accum_loss / max(global_step, 1) < best_loss:
                        best_loss = accum_loss / max(global_step, 1)

                if global_step >= total_steps:
                    break
            if global_step >= total_steps:
                break

        # ==============================================================
        # Step 7: 保存
        # ==============================================================
        progress_cb(96, "💾 保存蒸馏后的学生模型...")
        student.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

        # 保存蒸馏元信息
        meta = {
            "forgex_type": "distillation",
            "teacher_model": config.teacher_model,
            "student_model": config.student_model,
            "config": config.to_dict(),
            "final_loss": round(best_loss, 6),
            "total_steps": global_step,
            "layer_mapping": layer_map if layer_map else None,
        }
        (out_dir / "forgex_distill_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8",
        )

        # 释放教师
        del teacher
        if projectors is not None:
            del projectors
        try:
            import gc; gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

        progress_cb(100, f"✅ 蒸馏完成: {out_dir} (loss={best_loss:.4f})")
        return str(out_dir)

    # ================================================================
    #  辅助函数
    # ================================================================

    def _compute_layer_mapping(
        self, student_layers: int, teacher_layers: int, strategy: str,
    ) -> List[Tuple[int, int]]:
        """计算学生-教师层映射

        策略:
          "auto"   — 等间距映射 (默认)
          "even"   — 同 auto
          "last_n" — 只映射学生最后 N 层到教师最后 N 层
          "1:2,3:6,5:10" — 手动指定 student_layer:teacher_layer
        """
        if strategy in ("auto", "even", ""):
            # 等间距: 学生每层映射到教师对应比例位置
            mapping = []
            for s in range(student_layers):
                t = int(round(s * (teacher_layers - 1) / max(student_layers - 1, 1)))
                mapping.append((s + 1, t + 1))  # +1 因为 hidden_states[0] 是 embedding
            return mapping

        if strategy == "last_n":
            n = min(student_layers, teacher_layers)
            return [
                (student_layers - n + i + 1, teacher_layers - n + i + 1)
                for i in range(n)
            ]

        # 手动: "1:2,3:6,5:10"
        mapping = []
        for pair in strategy.split(","):
            pair = pair.strip()
            if ":" in pair:
                s, t = pair.split(":", 1)
                mapping.append((int(s.strip()), int(t.strip())))
        return mapping

    def _init_sparse_masks(
        self, model, target_sparsity: float, device, dtype,
    ) -> Dict[str, Any]:
        """基于权重幅值初始化稀疏掩码

        原理: L1 范数最小的神经元贡献最少，优先关闭
        """
        masks = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # 只对 MLP 层做稀疏（gate_proj, up_proj, down_proj）
            if any(k in name for k in ["gate_proj", "up_proj", "down_proj"]):
                # 按 L1 范数排序，标记最弱的 N%
                with torch.no_grad():
                    importance = param.abs().sum(dim=-1)  # per-output-neuron importance
                    masks[name] = {
                        "importance": importance,
                        "shape": param.shape,
                    }
        return masks

    def _apply_sparse_masks(
        self, model, masks: Dict, current_sparsity: float,
    ):
        """应用稀疏掩码: 将最弱的神经元梯度置零"""
        for name, param in model.named_parameters():
            if name in masks and param.grad is not None:
                importance = masks[name]["importance"]
                k = int(current_sparsity * importance.numel())
                if k > 0:
                    _, indices = importance.topk(k, largest=False)
                    param.grad[indices] = 0

    def _load_dataset(self, path: Path, tokenizer, max_seq_len: int):
        """加载并 tokenize 数据集"""
        import torch
        from datasets import Dataset

        rows = []
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            for line in text.strip().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        elif path.suffix == ".json":
            data = json.loads(text)
            rows = data if isinstance(data, list) else [data]
        else:
            # 纯文本
            rows = [{"text": text}]

        # 构建 prompt
        texts = []
        for r in rows:
            if "instruction" in r and "output" in r:
                inp = r.get("input", "")
                prompt = r["instruction"]
                if inp:
                    prompt += f"\n{inp}"
                full = f"{prompt}\n{r['output']}"
                texts.append(full)
            elif "text" in r:
                texts.append(str(r["text"]))
            elif "messages" in r or "conversations" in r:
                msgs = r.get("messages") or r.get("conversations") or []
                try:
                    full = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=False,
                    )
                    texts.append(full)
                except Exception:
                    texts.append(" ".join(m.get("content", "") for m in msgs if isinstance(m, dict)))

        if not texts:
            raise ValueError("数据集为空或格式不支持")

        # Tokenize
        all_input_ids = []
        all_attention = []
        all_labels = []
        for t in texts:
            enc = tokenizer(
                t, truncation=True, max_length=max_seq_len,
                padding="max_length", return_tensors="pt",
            )
            ids = enc["input_ids"].squeeze(0)
            mask = enc["attention_mask"].squeeze(0)
            labels = ids.clone()
            labels[mask == 0] = -100
            all_input_ids.append(ids)
            all_attention.append(mask)
            all_labels.append(labels)

        return Dataset.from_dict({
            "input_ids": torch.stack(all_input_ids),
            "attention_mask": torch.stack(all_attention),
            "labels": torch.stack(all_labels),
        }).with_format("torch")

    def _build_optimizer(self, params, lr, device):
        import torch
        try:
            if device == "cuda":
                from bitsandbytes.optim import PagedAdamW8bit
                return PagedAdamW8bit(params, lr=lr, weight_decay=0.01)
        except ImportError:
            pass
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    def _build_scheduler(self, optimizer, total_steps):
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))


# 单例
distiller = DistillationTrainer()


# ================================================================
#  API 教师蒸馏 — 用 API（OpenAI 兼容 / DeepSeek / Claude）生成教师数据
# ================================================================

@dataclass
class APITeacherConfig:
    """API 教师配置"""
    api_base: str = "https://api.openai.com/v1"  # OpenAI 兼容端点
    api_key: str = ""
    teacher_model_name: str = "gpt-4o-mini"       # API 模型名
    student_model: str = ""                        # 学生模型（本地）
    dataset_path: str = ""                         # 种子数据（instruction 列表）
    output_name: str = "api_distilled"

    # 生成参数
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 2048
    system_prompt: str = ""                        # 教师 system prompt

    # 训练参数
    lr: float = 2e-4
    batch_size: int = 1
    epochs: float = 3.0
    max_seq_len: int = 2048
    rank: int = 64
    gradient_accumulation_steps: int = 4
    use_qlora: bool = False

    # 增强
    num_responses: int = 1           # 每条 prompt 生成几个教师回复
    filter_quality: bool = True      # 过滤低质量回复（按长度）

    # 效率优化
    workers: int = 5                 # 并行 API 调用数（1=串行，推荐 5-10）
    pipeline_mode: bool = False      # 流水线模式：边生成数据边训练
    pipeline_batch_size: int = 500   # 流水线：每积累多少条开始训练一轮
    checkpoint_interval: int = 100   # 每生成多少条保存一次检查点

    # 做题学习模式
    exam_mode: str = "generate"      # "generate"=传统生成 | "exam"=做题学习 | "exam_dpo"=做题+DPO
    exam_rounds: int = 1             # 每轮: API 出新题 → 学生答 → 教师改 → 训练
    questions_per_round: int = 50    # 每轮 API 生成多少道新题（推荐 30-200）
    weak_focus_ratio: float = 2.0    # 错题重复倍率

    # ═══ 增强模式（从根源提升训练质量）═══
    # ① 思维链蒸馏: 教师不只给答案，给完整推理过程 → 学生学会"怎么想"
    cot_distill: bool = True

    # ② 多答案对比: 学生每题生成N个答案 → 教师选最好+最差 → 更丰富的DPO对
    best_of_n: int = 1              # 1=关闭, 3=推荐（每题3个答案选优劣）

    # ③ 渐进式容量增长: 每轮自动提高 LoRA rank → 模型容量逐步扩大
    progressive_rank: bool = False   # True=每轮 rank 递增（如 32→48→64→80）

    # ④ 主题专家路由: 按主题训练专门LoRA → 相当于MoE神经元切换
    topic_expert: bool = False       # True=启用主题专家（每轮额外训练分类LoRA）
    topic_expert_k: int = 4          # 专家数量（主题分类数）

    # ═══ 知识膨胀（训练前自动扩展学生模型参数）═══
    # 在蒸馏/做题之前先给学生模型"长身体"，增加参数容量再灌知识
    expand_before_train: bool = False  # True=训练前先膨胀学生模型
    expand_method: str = "depth"       # "depth" | "width" | "hybrid" | "depth+width"
    expand_extra_layers: int = 4       # 深度膨胀: 加几层
    expand_depth_strategy: str = "repeat_middle"  # 深度策略
    expand_target_hidden: int = 0      # 宽度膨胀: 目标 hidden (0=不扩宽)
    expand_target_intermediate: int = 0  # 宽度: 目标 MLP (0=自动)
    expand_noise: float = 0.01         # 膨胀噪声系数

    # ═══ 训练后浓缩（锻压: 剪枝 + 自蒸馏 + 困难精修）═══
    condense_after_train: bool = False  # True=训练后自动浓缩
    condense_cycles: int = 1            # 锻压循环次数 (1=单次, 2-3=深度淬火)
    condense_prune_heads: float = 0.2   # 剪掉多少 attention head (0~0.5)
    condense_prune_neurons: float = 0.15  # 剪掉多少 MLP neuron (0~0.5)
    condense_prune_layers: int = 0      # 剪掉几层 (0=不剪层)
    condense_self_distill: bool = True  # 是否做自蒸馏浓缩
    condense_distill_temp: float = 3.0  # 蒸馏温度
    condense_hard_mine: bool = True     # 是否挖掘困难样本再训练
    condense_hard_ratio: float = 0.3    # 困难样本比例

    def to_dict(self) -> Dict:
        d = {k: v for k, v in self.__dict__.items()}
        d.pop("api_key", None)  # 不保存 API key
        return d


class APITeacherDistiller:
    """用 API 作为教师，生成高质量训练数据 → SFT 训练学生模型

    工作流:
      1. (可选) 膨胀学生模型 — 增加参数容量
      2. 读取种子 prompt（用户已有的 instruction 数据集）
      3. 调用 API 生成教师回复（soft teacher output）
      4. 将教师数据保存为 SFT 格式
      5. 用教师数据 SFT 训练学生模型

    支持的 API:
      - OpenAI / GPT-4o / GPT-4o-mini
      - DeepSeek
      - Claude (通过 OpenAI 兼容代理)
      - 任何 OpenAI 兼容 API
      - 本地 vLLM / Ollama（兼容 OpenAI 格式）
    """

    def _maybe_expand_student(self, config: APITeacherConfig, task=None) -> str:
        """如果启用了知识膨胀，先扩展学生模型参数再训练

        返回: 实际用于训练的模型路径（膨胀后的或原始的）
        """
        if not config.expand_before_train:
            return config.student_model

        progress_cb = task.update_progress if task else lambda p, m="": log(m)
        method = config.expand_method
        student = config.student_model

        expanded_name = f"{config.output_name}_expanded"
        log(f"🔧 知识膨胀: {method} | 学生模型 = {student}")
        progress_cb(1, f"💪 训练前膨胀: {method}...")

        try:
            from core.expansion import depth_expand, width_expand, hybrid_expand

            if method == "depth":
                result = depth_expand(
                    student, expanded_name,
                    strategy=config.expand_depth_strategy,
                    num_new_layers=config.expand_extra_layers,
                    noise_scale=config.expand_noise,
                    task=task,
                )
            elif method == "width":
                result = width_expand(
                    student, expanded_name,
                    target_hidden=config.expand_target_hidden,
                    target_intermediate=config.expand_target_intermediate,
                    noise_scale=config.expand_noise,
                    task=task,
                )
            elif method in ("hybrid", "depth+width"):
                # 先估算目标层数
                from transformers import AutoConfig
                try:
                    cfg = AutoConfig.from_pretrained(student, trust_remote_code=True)
                    orig_layers = cfg.num_hidden_layers
                except Exception:
                    orig_layers = 12
                target_layers = orig_layers + config.expand_extra_layers

                result = hybrid_expand(
                    student, expanded_name,
                    target_layers=target_layers,
                    target_hidden=config.expand_target_hidden,
                    target_intermediate=config.expand_target_intermediate,
                    depth_strategy=config.expand_depth_strategy,
                    noise_scale=config.expand_noise,
                    task=task,
                )
            else:
                log(f"⚠️ 未知膨胀方式: {method}，跳过")
                return student

            log(f"✅ 膨胀完成: {result}")
            progress_cb(10, f"✅ 膨胀完成 → 使用扩展后模型训练")

            # 更新 config 指向膨胀后的模型
            config.student_model = str(result)
            return str(result)

        except Exception as e:
            log(f"⚠️ 膨胀失败: {e}，使用原始模型继续")
            progress_cb(10, f"⚠️ 膨胀失败: {e}，跳过膨胀")
            return student

    def _maybe_condense(self, config: APITeacherConfig, trained_result: str, task=None) -> str:
        """训练后自动浓缩: 剪枝 → 自蒸馏 → 困难精修

        把"虚胖"的训练结果压成"精瘦"的高密度模型。
        """
        if not config.condense_after_train:
            return trained_result

        progress_cb = task.update_progress if task else lambda p, m="": log(m)
        log(f"🔧 训练后浓缩启动: cycles={config.condense_cycles}")
        progress_cb(82, "⚗️ 开始知识浓缩...")

        try:
            from core.condenser import structured_prune, self_distill, mine_hard_examples

            current = trained_result
            cal_data = config.dataset_path  # 用训练数据做校准
            base_name = config.output_name

            for ci in range(max(1, config.condense_cycles)):
                suffix = f"_c{ci+1}"
                progress_cb(82 + 15 * ci / config.condense_cycles,
                             f"浓缩第 {ci+1}/{config.condense_cycles} 轮...")

                # Step 1: 剪枝
                if config.condense_prune_heads > 0 or config.condense_prune_neurons > 0:
                    pruned_name = f"{base_name}{suffix}_pruned"
                    try:
                        current = structured_prune(
                            current, pruned_name,
                            calibration_data=cal_data,
                            head_prune_ratio=config.condense_prune_heads,
                            neuron_prune_ratio=config.condense_prune_neurons,
                            layer_prune_count=config.condense_prune_layers if ci == config.condense_cycles - 1 else 0,
                        )
                        log(f"  剪枝完成: {current}")
                    except Exception as e:
                        log(f"  剪枝跳过: {e}")

                # Step 2: 自蒸馏
                if config.condense_self_distill:
                    distilled_name = f"{base_name}{suffix}_distilled"
                    try:
                        current = self_distill(
                            teacher_path=trained_result,  # 原始训练结果当 teacher
                            student_path=current,          # 剪枝后的当 student
                            calibration_data=cal_data,
                            output_name=distilled_name,
                            temperature=config.condense_distill_temp,
                        )
                        log(f"  自蒸馏完成: {current}")
                    except Exception as e:
                        log(f"  自蒸馏跳过: {e}")

                # Step 3: 困难精修
                if config.condense_hard_mine:
                    hard_name = f"{base_name}{suffix}_hard"
                    try:
                        hard_path = mine_hard_examples(
                            current, cal_data,
                            output_name=hard_name,
                            top_ratio=config.condense_hard_ratio,
                        )
                        # 用困难样本做一轮 SFT 精修
                        from core.trainer import trainer_engine
                        refined_name = f"{base_name}{suffix}_refined"
                        trainer_engine.train(
                            method="sft", backend="auto",
                            base_model=current,
                            dataset_path=hard_path,
                            params={
                                "output_name": refined_name,
                                "lr": 5e-5, "batch_size": 1, "epochs": 1,
                                "max_seq_len": 1024, "rank": 32,
                            },
                        )
                        current = str(Path(LORAS_DIR) / refined_name)
                        log(f"  困难精修完成: {current}")
                    except Exception as e:
                        log(f"  困难精修跳过: {e}")

            progress_cb(97, f"⚗️ 浓缩完成")
            return current

        except Exception as e:
            log(f"⚠️ 浓缩失败: {e}，返回原始结果")
            return trained_result

    def generate_teacher_data(self, config: APITeacherConfig, task=None) -> Path:
        """Step 1: 并行调用 API 生成教师数据（支持断点续传）"""
        import concurrent.futures
        import threading
        import time as _time

        progress_cb = task.update_progress if task else lambda p, m="": log(m)

        progress_cb(2, "📋 加载种子数据...")
        prompts = self._load_prompts(config)
        if not prompts:
            raise ValueError("种子数据为空。请提供包含 instruction 字段的 JSON/JSONL 文件。")
        progress_cb(5, f"✅ 种子数据: {len(prompts)} 条 prompt")

        # 输出目录
        out_dir = Path(DATASETS_DIR) / f"{config.output_name}_teacher"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "teacher_data.json"
        checkpoint_file = out_dir / "_checkpoint.jsonl"

        # ---- 断点续传：检查已有进度 ----
        existing_data = []
        completed_indices = set()
        if checkpoint_file.exists():
            try:
                for line in checkpoint_file.read_text(encoding="utf-8").strip().splitlines():
                    if line.strip():
                        item = json.loads(line)
                        idx = item.pop("_idx", None)
                        if idx is not None:
                            completed_indices.add(idx)
                        existing_data.append(item)
                if existing_data:
                    progress_cb(6, f"📌 发现检查点: 已有 {len(existing_data)} 条，跳过已完成")
                    log(f"断点续传: {len(existing_data)} 条已有数据，{len(completed_indices)} 个 prompt 已完成")
            except Exception:
                existing_data = []
                completed_indices = set()

        # ---- 构建待处理列表 ----
        work_items = []
        for i, prompt_item in enumerate(prompts):
            if i in completed_indices:
                continue
            instruction = prompt_item.get("instruction", "")
            input_text = prompt_item.get("input", "")
            if not instruction:
                continue
            for r in range(config.num_responses):
                work_items.append((i, instruction, input_text, r))

        total_work = len(work_items) + len(existing_data)
        if not work_items:
            if existing_data:
                progress_cb(70, f"✅ 所有数据已在检查点中 ({len(existing_data)} 条)")
                self._save_teacher_data(existing_data, out_file, out_dir, config, prompts)
                return out_file
            raise RuntimeError("没有有效的 prompt 需要处理")

        progress_cb(7, f"📊 待处理: {len(work_items)} 条 | 已完成: {len(existing_data)} 条")

        # ---- 构建 API 客户端 ----
        progress_cb(8, f"🔗 连接 API: {config.api_base}")
        client = self._make_client(config)

        # 测试连接
        progress_cb(9, "🔗 测试 API 连接...")
        try:
            client.chat([{"role": "user", "content": "hi"}], temperature=0, max_tokens=5)
            progress_cb(10, f"✅ API 连接成功 (模型: {config.teacher_model_name})")
        except Exception as e:
            raise RuntimeError(f"API 连接测试失败: {e}\n请检查 API Key、Base URL 和模型名称。") from e

        # ---- 并行生成 ----
        workers = max(1, min(config.workers, 20))  # 限制 1-20
        teacher_data = list(existing_data)  # 从检查点恢复
        lock = threading.Lock()
        done_count = len(existing_data)
        error_count = 0
        start_time = _time.time()

        # 打开 checkpoint 文件（追加模式）
        ckpt_f = open(str(checkpoint_file), "a", encoding="utf-8")

        def _process_one(item):
            nonlocal done_count, error_count
            idx, instruction, input_text, resp_idx = item
            try:
                reply = self._call_api(client, config, instruction=instruction, input_text=input_text)
                if reply and reply.strip():
                    if config.filter_quality and len(reply.strip()) < 10:
                        return None
                    result = {"instruction": instruction, "output": reply.strip()}
                    if input_text:
                        result["input"] = input_text
                    # 写入 checkpoint（线程安全）
                    with lock:
                        teacher_data.append(result)
                        ckpt_line = json.dumps({**result, "_idx": idx}, ensure_ascii=False, default=str)
                        ckpt_f.write(ckpt_line + "\n")
                        ckpt_f.flush()
                        done_count += 1
                    return result
            except Exception as e:
                with lock:
                    error_count += 1
                    done_count += 1
                log(f"  API [{idx}] 失败: {e}")
                return None

        progress_cb(11, f"🚀 开始生成 (并行×{workers})...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_one, item): item for item in work_items}

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # 触发异常传播
                except Exception:
                    pass

                # 更新进度
                with lock:
                    current_done = done_count
                    current_collected = len(teacher_data)
                    current_errors = error_count

                if current_done % max(1, min(10, len(work_items) // 50)) == 0 or current_done == total_work:
                    elapsed = _time.time() - start_time
                    speed = current_done / max(elapsed, 0.1)
                    remaining = (total_work - current_done) / max(speed, 0.01)
                    pct = 10 + int(60 * current_done / max(total_work, 1))
                    eta_str = f"{int(remaining)}s" if remaining < 120 else f"{int(remaining/60)}m"
                    progress_cb(pct, f"生成: {current_done}/{total_work} | ✅{current_collected} ❌{current_errors} | {speed:.1f}条/s | ETA {eta_str}")

        ckpt_f.close()

        if not teacher_data:
            raise RuntimeError("API 未返回任何有效数据。请检查 API Key 和端点。")

        # ---- 保存最终数据 ----
        self._save_teacher_data(teacher_data, out_file, out_dir, config, prompts)

        # 清理 checkpoint
        try:
            checkpoint_file.unlink()
        except Exception:
            pass

        elapsed_total = _time.time() - start_time
        progress_cb(75, f"✅ 教师数据: {len(teacher_data)} 条 | 耗时 {int(elapsed_total)}s | {len(teacher_data)/max(elapsed_total,0.1):.1f}条/s")
        return out_file

    def _save_teacher_data(self, data, out_file, out_dir, config, prompts):
        """保存教师数据和元信息"""
        out_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        meta = {
            "forgex_type": "api_teacher_data",
            "api_model": config.teacher_model_name,
            "api_base": config.api_base,
            "total_prompts": len(prompts),
            "total_responses": len(data),
            "config": config.to_dict(),
        }
        (out_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8",
        )

    def train_student(self, config: APITeacherConfig, teacher_data_path: Path = None, task=None) -> str:
        """训练学生模型

        支持三种模式:
        - generate (传统): 教师生成数据 → SFT 训练
        - exam (做题学习): 学生做题 → 教师批改 → 错题加权SFT
        - exam_dpo (做题+DPO): 做题学习 + 生成 DPO 偏好数据

        如果启用了 expand_before_train，会在训练前先膨胀学生模型参数。
        """
        progress_cb = task.update_progress if task else lambda p, m="": log(m)

        # ---- 知识膨胀（可选）----
        self._maybe_expand_student(config, task)

        # ---- 做题学习模式 ----
        if config.exam_mode in ("exam", "exam_dpo"):
            result = self.run_exam_learning(config, task)
            result = self._maybe_condense(config, str(result), task)
            return result

        # ---- 传统生成模式 ----

        # ---- 普通模式: 先全部生成，再训练 ----
        if not config.pipeline_mode:
            if teacher_data_path is None or not Path(teacher_data_path).exists():
                progress_cb(2, "📡 先生成教师数据...")
                teacher_data_path = self.generate_teacher_data(config, task)

            progress_cb(78, "🎓 开始用教师数据训练学生模型...")
            from core.trainer import trainer_engine
            result = trainer_engine.train(
                method="sft",
                backend="auto",
                base_model=config.student_model,
                dataset_path=str(teacher_data_path),
                params={
                    "output_name": config.output_name,
                    "lr": config.lr,
                    "batch_size": config.batch_size,
                    "epochs": config.epochs,
                    "max_seq_len": config.max_seq_len,
                    "rank": config.rank,
                    "gradient_accumulation_steps": config.gradient_accumulation_steps,
                    "use_qlora": config.use_qlora,
                },
                task=task,
            )
            progress_cb(80, f"✅ API 蒸馏完成: {result}")
            result = self._maybe_condense(config, str(result), task)
            progress_cb(100, f"✅ API 蒸馏+浓缩完成: {result}")
            return result

        # ---- 流水线模式: 分批生成 + 增量训练 ----
        import concurrent.futures
        import threading
        import time as _time

        progress_cb(2, "🔄 流水线模式: 边生成数据边训练")

        prompts = self._load_prompts(config)
        if not prompts:
            raise ValueError("种子数据为空")
        progress_cb(5, f"✅ 种子数据: {len(prompts)} 条")

        client = self._make_client(config)
        # 测试连接
        try:
            client.chat([{"role": "user", "content": "hi"}], temperature=0, max_tokens=5)
            progress_cb(8, f"✅ API 连接成功")
        except Exception as e:
            raise RuntimeError(f"API 连接失败: {e}") from e

        # 分批
        batch_size = max(50, config.pipeline_batch_size)
        batches = []
        for i in range(0, len(prompts), batch_size):
            batches.append(prompts[i:i + batch_size])

        out_dir = Path(DATASETS_DIR) / f"{config.output_name}_teacher"
        out_dir.mkdir(parents=True, exist_ok=True)
        accumulated_data = []
        train_round = 0
        last_result = ""
        workers = max(1, min(config.workers, 20))

        from core.trainer import trainer_engine

        for bi, batch in enumerate(batches):
            # ---- 生成当前批次 ----
            progress_cb(
                10 + int(70 * bi / len(batches)),
                f"📡 批次 {bi+1}/{len(batches)}: 生成 {len(batch)} 条 (并行×{workers})"
            )

            batch_data = []
            lock = threading.Lock()

            def _gen_one(prompt_item):
                instruction = prompt_item.get("instruction", "")
                input_text = prompt_item.get("input", "")
                if not instruction:
                    return None
                for _ in range(config.num_responses):
                    try:
                        reply = self._call_api(client, config, instruction=instruction, input_text=input_text)
                        if reply and reply.strip() and (not config.filter_quality or len(reply.strip()) >= 10):
                            result = {"instruction": instruction, "output": reply.strip()}
                            if input_text:
                                result["input"] = input_text
                            with lock:
                                batch_data.append(result)
                            return result
                    except Exception as e:
                        log(f"  API 失败: {e}")
                return None

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                list(executor.map(_gen_one, batch))

            if not batch_data:
                log(f"  批次 {bi+1} 无有效数据，跳过")
                continue

            accumulated_data.extend(batch_data)

            # 保存累积数据
            data_file = out_dir / "teacher_data.json"
            data_file.write_text(
                json.dumps(accumulated_data, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

            # ---- 增量训练 ----
            train_round += 1
            train_epochs = max(0.5, config.epochs / len(batches))  # 每轮少训一点
            progress_cb(
                10 + int(70 * (bi + 0.5) / len(batches)),
                f"🎓 训练轮 {train_round}: {len(accumulated_data)} 条数据 × {train_epochs:.1f} epochs"
            )

            try:
                last_result = trainer_engine.train(
                    method="sft",
                    backend="auto",
                    base_model=config.student_model,
                    dataset_path=str(data_file),
                    params={
                        "output_name": f"{config.output_name}_round{train_round}",
                        "lr": config.lr,
                        "batch_size": config.batch_size,
                        "epochs": train_epochs,
                        "max_seq_len": config.max_seq_len,
                        "rank": config.rank,
                        "gradient_accumulation_steps": config.gradient_accumulation_steps,
                        "use_qlora": config.use_qlora,
                    },
                    task=task,
                )
                log(f"  训练轮 {train_round} 完成: {last_result}")
            except Exception as e:
                log(f"  训练轮 {train_round} 失败: {e}")

        # 最终保存元信息
        self._save_teacher_data(accumulated_data, out_dir / "teacher_data.json", out_dir, config, prompts)

        progress_cb(85, f"✅ 流水线蒸馏完成: {len(accumulated_data)} 条数据, {train_round} 轮训练")
        last_result = self._maybe_condense(config, str(last_result), task)
        progress_cb(100, f"✅ 流水线蒸馏+浓缩完成")
        return last_result

    # ================================================================
    #  做题学习模式 — 学生做题 → 教师批改 → 针对性训练
    # ================================================================

    def run_exam_learning(self, config: APITeacherConfig, task=None) -> str:
        """做题学习模式: 逐题实时 — 学生答一题 → 教师改一题 → 实时显示 → 下一题

        每轮: API出新题 → 逐题(答+改+显示) → 训练 → 下轮用改进模型
        """
        import time as _time

        progress_cb = task.update_progress if task else lambda p, m="": log(m)
        progress_cb(1, "📋 做题学习模式启动")

        # ═══ 预检查: 学生模型路径有效性 ═══
        student_p = Path(config.student_model)
        if student_p.is_dir():
            has_config = (student_p / "config.json").exists()
            has_adapter = (student_p / "adapter_config.json").exists()
            if not has_config and not has_adapter:
                raise ValueError(
                    f"学生模型路径无效: {config.student_model}\n"
                    f"目录中没有 config.json 或 adapter_config.json。\n"
                    f"请指定正确的 HuggingFace 模型路径或 LoRA 适配器路径。"
                )
            if has_adapter and not has_config:
                # 检查 LoRA 的基座模型是否可达
                try:
                    base_test, _ = self._resolve_base_model(config.student_model)
                    progress_cb(1, f"📋 检测到 LoRA 适配器，基座: {Path(base_test).name}")
                except ValueError as e:
                    raise ValueError(f"学生模型基座路径问题:\n{e}") from e
        elif not student_p.exists() and "/" not in config.student_model:
            raise ValueError(
                f"学生模型不存在: {config.student_model}\n"
                f"请填写有效的本地路径或 HuggingFace 模型ID（如 Qwen/Qwen2.5-1.5B-Instruct）"
            )

        # ═══ VRAM 状态 ═══
        _vram_log("做题模式启动")

        # 加载种子数据
        seed_prompts = self._load_prompts(config)
        if not seed_prompts:
            raise ValueError("种子数据为空")
        progress_cb(3, f"✅ 种子数据: {len(seed_prompts)} 条（用于提取主题方向）")

        # 连接 API 教师
        client = self._make_client(config)
        try:
            client.chat([{"role": "user", "content": "hi"}], temperature=0, max_tokens=5)
            progress_cb(5, f"✅ 教师 API 连接成功 ({config.teacher_model_name})")
        except Exception as e:
            raise RuntimeError(f"教师 API 连接失败: {e}") from e

        out_dir = Path(DATASETS_DIR) / f"{config.output_name}_exam"
        out_dir.mkdir(parents=True, exist_ok=True)

        all_sft_data = []
        all_dpo_data = []
        round_scores = []
        current_model = config.student_model

        from core.trainer import trainer_engine

        total_rounds = max(1, config.exam_rounds)
        for exam_round in range(1, total_rounds + 1):
            round_start = _time.time()
            base_pct = 5 + int(85 * (exam_round - 1) / total_rounds)
            step_pct = int(85 / total_rounds)

            # ═══════ Step 1: API 出新题 ═══════
            n_questions = max(5, config.questions_per_round)
            progress_cb(base_pct, f"📝 R{exam_round}: API 出 {n_questions} 道新题...")
            new_questions = self._generate_exam_questions(
                client, config, seed_prompts, n_questions, exam_round, task,
                base_pct=base_pct, pct_range=int(step_pct * 0.1),
            )
            if not new_questions:
                log(f"  R{exam_round}: API 出题失败，用种子数据")
                import random
                new_questions = random.sample(seed_prompts, min(n_questions, len(seed_prompts)))
            progress_cb(base_pct + int(step_pct * 0.1),
                        f"✅ R{exam_round}: 出题 {len(new_questions)} 道")

            # ═══════ Step 2: 加载学生模型 ═══════
            progress_cb(base_pct + int(step_pct * 0.12),
                        f"🎓 R{exam_round}: 加载学生模型...")
            _vram_log(f"  R{exam_round} 加载前")
            student_model, student_tok = self._load_student_model(current_model)

            if student_model is None:
                # ═══ 模型加载失败: 跳过本轮而非崩溃 ═══
                err_msg = f"R{exam_round}: 学生模型加载失败 ({Path(current_model).name})"
                log(f"  ❌ {err_msg}")
                if task:
                    task.logs.append(f"❌ {err_msg}")
                    task.logs.append(f"  提示: 检查模型路径是否正确，或显存是否足够")
                    task.logs.append(f'[METRIC]{json.dumps({"type":"exam_round","round":exam_round,"avg":0,"correct":0,"wrong":0,"total":0,"error":"model_load_failed"}, ensure_ascii=False, default=str)}')
                round_scores.append({"round": exam_round, "avg": 0, "correct": 0,
                                     "wrong": 0, "total": 0, "error": "model_load_failed"})
                # 退回原始基座模型重试
                if current_model != config.student_model:
                    log(f"  🔄 退回原始基座模型: {Path(config.student_model).name}")
                    current_model = config.student_model
                continue

            # ═══════ Step 3: 逐题 — 答题+批改+实时显示 ═══════
            graded = []
            total_q = len(new_questions)
            cum_score = 0
            bon = max(1, config.best_of_n)  # Best-of-N

            for qi, q_data in enumerate(new_questions):
                instruction = q_data.get("instruction", "")
                input_text = q_data.get("input", "")
                if not instruction:
                    continue

                # ---- Best-of-N: 学生生成多个答案 ----
                if bon > 1 and student_model is not None:
                    candidates = []
                    for ni in range(bon):
                        ans = self._student_answer_one(student_model, student_tok, instruction, input_text)
                        candidates.append(ans)

                    # 教师只回答一次（避免重复 API 调用）
                    teacher_answer = self._get_teacher_answer(client, config, instruction, input_text)

                    # 用同一个标准答案评判每个候选
                    best_score, best_idx, worst_score, worst_idx = 0, 0, 6, 0
                    candidate_scores = []
                    for ci, cand in enumerate(candidates):
                        sc = self._grade_against_answer(client, config, instruction, input_text, teacher_answer, cand)
                        candidate_scores.append(sc)
                        if sc > best_score:
                            best_score, best_idx = sc, ci
                        if sc < worst_score:
                            worst_score, worst_idx = sc, ci

                    student_ans = candidates[best_idx]
                    grade = {
                        "instruction": instruction, "input": input_text,
                        "student_answer": student_ans,
                        "teacher_answer": teacher_answer,
                        "score": best_score, "weakness": "",
                    }

                    # 如果有明显差异，生成 DPO 对（best vs worst）
                    if best_score > worst_score and best_score >= 3:
                        grade["_bon_best"] = candidates[best_idx]
                        grade["_bon_worst"] = candidates[worst_idx]
                        grade["_bon_best_score"] = best_score
                        grade["_bon_worst_score"] = worst_score
                else:
                    # ---- 单答案模式 ----
                    student_ans = self._student_answer_one(student_model, student_tok, instruction, input_text)
                    grade = self._teacher_grade_one(client, config, instruction, input_text, student_ans)

                graded.append(grade)

                score = grade.get("score", 0)
                cum_score += score
                avg_now = cum_score / (qi + 1)

                # ---- 实时显示 ----
                if task:
                    qi_display = qi + 1
                    s_icon = "✅" if score >= 4 else "⚠️" if score == 3 else "❌"
                    task.logs.append(f"{'─'*50}")
                    bon_tag = f" [BoN:{bon}]" if bon > 1 else ""
                    cot_tag = " [CoT]" if config.cot_distill else ""
                    task.logs.append(f"📝 R{exam_round} Q{qi_display}/{total_q} | {s_icon} {score}/5 | 均分 {avg_now:.1f}{bon_tag}{cot_tag}")
                    task.logs.append(f"  题: {instruction[:80]}{'...' if len(instruction)>80 else ''}")
                    if student_ans:
                        task.logs.append(f"  答: {student_ans[:80]}{'...' if len(student_ans)>80 else ''}")
                    else:
                        task.logs.append(f"  答: (未作答)")
                    teacher_ans = grade.get("teacher_answer", "")
                    if teacher_ans and score < 4:
                        task.logs.append(f"  正: {teacher_ans[:80]}{'...' if len(teacher_ans)>80 else ''}")
                    weakness = grade.get("weakness", "")
                    if weakness:
                        task.logs.append(f"  评: {weakness}")

                    # METRIC for chart
                    task.logs.append(f'[METRIC]{json.dumps({"type":"exam_q","round":exam_round,"qi":qi,"score":score,"avg":round(avg_now,2),"answered":bool(student_ans),"total":total_q}, ensure_ascii=False, default=str)}')

                # 进度条
                q_pct = base_pct + int(step_pct * (0.15 + 0.5 * (qi + 1) / total_q))
                progress_cb(q_pct, f"📊 R{exam_round} Q{qi+1}/{total_q} | 分:{score} 均:{avg_now:.1f}")

            # ═══════ 释放学生模型 ═══════
            # 必须在调用者侧清除引用，否则 Python GC 不会释放 GPU 内存
            del student_model, student_tok
            self._force_gc()

            # ═══════ Step 4: 统计本轮成绩 ═══════
            scores = [g["score"] for g in graded if "score" in g]
            avg_score = sum(scores) / max(len(scores), 1)
            score_dist = {s: sum(1 for x in scores if x == s) for s in range(1, 6)}
            correct = sum(1 for s in scores if s >= 4)
            wrong = sum(1 for s in scores if s < 4)
            unanswered = sum(1 for g in graded if not g.get("student_answer"))

            round_info = {
                "round": exam_round, "avg": round(avg_score, 2),
                "correct": correct, "wrong": wrong,
                "unanswered": unanswered, "total": len(scores),
                "distribution": score_dist,
            }
            round_scores.append(round_info)

            if task:
                task.logs.append(f'[METRIC]{json.dumps({"type":"exam_round","round":exam_round,"avg":round(avg_score,2),"correct":correct,"wrong":wrong,"total":len(scores),"distribution":score_dist}, ensure_ascii=False, default=str)}')

            progress_cb(base_pct + int(step_pct * 0.68),
                        f"📊 R{exam_round} 完成: 均分{avg_score:.1f} ✅{correct} ❌{wrong}")

            # ═══════ Step 5: 生成训练数据 ═══════
            round_sft = []
            round_dpo = []
            for g in graded:
                teacher_ans = g.get("teacher_answer", "")
                student_ans = g.get("student_answer", "")
                sc = g.get("score", 3)
                if not teacher_ans:
                    continue
                sft_item = {"instruction": g["instruction"], "output": teacher_ans}
                if g.get("input"):
                    sft_item["input"] = g["input"]
                if sc <= 2:
                    for _ in range(int(config.weak_focus_ratio)):
                        round_sft.append(dict(sft_item))
                else:
                    round_sft.append(sft_item)

                # DPO 偏好对 —— 两个来源
                if config.exam_mode == "exam_dpo":
                    # 来源1: Best-of-N 产生的自然对比（最佳学生答案 vs 最差学生答案）
                    bon_best = g.get("_bon_best")
                    bon_worst = g.get("_bon_worst")
                    if bon_best and bon_worst and bon_best != bon_worst:
                        dpo_item = {"instruction": g["instruction"],
                                    "chosen": bon_best, "rejected": bon_worst}
                        if g.get("input"):
                            dpo_item["input"] = g["input"]
                        round_dpo.append(dpo_item)

                    # 来源2: 教师答案 vs 学生错答
                    if sc <= 3 and student_ans:
                        dpo_item = {"instruction": g["instruction"],
                                    "chosen": teacher_ans, "rejected": student_ans}
                        if g.get("input"):
                            dpo_item["input"] = g["input"]
                        round_dpo.append(dpo_item)

            all_sft_data.extend(round_sft)
            all_dpo_data.extend(round_dpo)

            # ═══════ Step 6: 训练（链式递进）═══════

            # 渐进式容量: 每轮自动提高 LoRA rank
            if config.progressive_rank:
                base_rank = max(16, config.rank // 2)
                round_rank = min(base_rank + (exam_round - 1) * 16, config.rank * 2)
                log(f"  📈 渐进 rank: {round_rank} (R{exam_round}/{total_rounds})")
            else:
                round_rank = config.rank

            train_ep = max(1.0, config.epochs)

            if round_sft:
                sft_file = out_dir / f"sft_round{exam_round}.json"
                sft_file.write_text(json.dumps(round_sft, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
                round_output = f"{config.output_name}_r{exam_round}"

                enhancements = []
                if config.cot_distill: enhancements.append("CoT")
                if config.best_of_n > 1: enhancements.append(f"BoN×{config.best_of_n}")
                if config.progressive_rank: enhancements.append(f"rank={round_rank}")
                enh_str = f" [{'+'.join(enhancements)}]" if enhancements else ""
                progress_cb(base_pct + int(step_pct * 0.72),
                            f"🎓 训练: {len(round_sft)}条 × {train_ep}ep{enh_str}")
                try:
                    trainer_engine.train(
                        method="sft", backend="auto",
                        base_model=current_model,
                        dataset_path=str(sft_file),
                        params={
                            "output_name": round_output,
                            "lr": config.lr, "batch_size": config.batch_size,
                            "epochs": train_ep, "max_seq_len": config.max_seq_len,
                            "rank": round_rank, "use_qlora": config.use_qlora,
                            "gradient_accumulation_steps": config.gradient_accumulation_steps,
                        },
                        task=task,
                    )
                    trained_path = Path("data/loras") / round_output
                    if trained_path.exists():
                        # ═══ 关键: 合并 LoRA → 完整模型，确保下一轮能正确加载 ═══
                        if total_rounds > 1:
                            progress_cb(base_pct + int(step_pct * 0.82),
                                        f"🔀 R{exam_round}: 合并 LoRA 到基座模型...")
                            merged = self._merge_lora_for_chain(
                                str(trained_path), f"R{exam_round}-SFT", task)
                            if merged:
                                current_model = merged
                                log(f"  ✅ 下轮用合并模型: {merged}")
                            else:
                                # 合并失败，退回原始基座（至少不会崩溃）
                                current_model = config.student_model
                                log(f"  ⚠️ 合并失败，下轮用原始基座模型")
                        else:
                            current_model = str(trained_path)
                            log(f"  ✅ 训练完成: {round_output}")
                except Exception as e:
                    log(f"  SFT 训练失败: {e}")

            if round_dpo and len(round_dpo) >= 3:
                dpo_file = out_dir / f"dpo_round{exam_round}.json"
                dpo_file.write_text(json.dumps(round_dpo, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
                dpo_output = f"{config.output_name}_dpo_r{exam_round}"
                bon_note = f" (含{sum(1 for g in graded if g.get('_bon_best'))}组BoN对)" if config.best_of_n > 1 else ""
                progress_cb(base_pct + int(step_pct * 0.9),
                            f"🎯 DPO: {len(round_dpo)}对{bon_note}")
                try:
                    trainer_engine.train(
                        method="dpo", backend="auto", base_model=current_model,
                        dataset_path=str(dpo_file),
                        params={
                            "output_name": dpo_output,
                            "lr": config.lr * 0.5, "batch_size": config.batch_size,
                            "epochs": max(1.0, train_ep * 0.5), "max_seq_len": config.max_seq_len,
                            "rank": round_rank, "use_qlora": config.use_qlora,
                            "gradient_accumulation_steps": config.gradient_accumulation_steps,
                        },
                        task=task,
                    )
                    dpo_path = Path("data/loras") / dpo_output
                    if dpo_path.exists():
                        # DPO 也需要合并
                        if total_rounds > 1 and exam_round < total_rounds:
                            merged_dpo = self._merge_lora_for_chain(
                                str(dpo_path), f"R{exam_round}-DPO", task)
                            if merged_dpo:
                                current_model = merged_dpo
                        else:
                            current_model = str(dpo_path)
                except Exception as e:
                    log(f"  DPO 失败: {e}")

            log(f"  R{exam_round} 完成 {int(_time.time()-round_start)}s | SFT {len(round_sft)} DPO {len(round_dpo)}")
            _vram_log(f"  R{exam_round} 完成后")

        # ═══════ 主题专家训练（MoLoRA 路由）═══════
        if config.topic_expert and all_sft_data:
            try:
                progress_cb(90, f"🧠 主题专家: 分类 {len(all_sft_data)} 条训练数据...")
                topic_loras = self._train_topic_experts(
                    client, config, all_sft_data, out_dir, current_model, task, trainer_engine)
                if topic_loras:
                    progress_cb(94, f"✅ 主题专家完成: {len(topic_loras)} 个专家 LoRA")
            except Exception as e:
                log(f"  主题专家训练失败（不影响主流程）: {e}")

        # ═══════ 最终合并: 确保输出一个完整可用的模型 ═══════
        final_model_path = current_model
        if total_rounds > 0:
            # 检查最终模型是否是 LoRA adapter（需要合并）
            base_check, lora_check = self._resolve_base_model(current_model)
            if lora_check:
                progress_cb(95, "🔀 最终合并: LoRA → 完整模型...")
                merged_final = self._merge_lora_for_chain(
                    current_model, "最终合并", task)
                if merged_final:
                    final_model_path = merged_final

            # 移动到最终输出位置（rename 而非 copy，避免复制 GB 级文件）
            final_output_dir = Path(LORAS_DIR) / f"{config.output_name}_final"
            src_path = Path(final_model_path)
            if src_path.is_dir() and src_path != final_output_dir:
                if final_output_dir.exists():
                    import shutil
                    shutil.rmtree(final_output_dir, ignore_errors=True)
                try:
                    src_path.rename(final_output_dir)
                    final_model_path = str(final_output_dir)
                    log(f"  ✅ 最终模型: {final_output_dir}")
                except OSError:
                    # 跨文件系统时 rename 失败，退回 copytree
                    try:
                        import shutil
                        shutil.copytree(str(src_path), str(final_output_dir))
                        final_model_path = str(final_output_dir)
                        log(f"  ✅ 最终模型(复制): {final_output_dir}")
                    except Exception as e:
                        log(f"  最终模型移动失败: {e}")

        # ═══════ 汇总训练数据集（方便用户复用）═══════
        if all_sft_data:
            consolidated_sft = out_dir / "all_sft_data.json"
            consolidated_sft.write_text(
                json.dumps(all_sft_data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            log(f"  📦 汇总 SFT 数据: {len(all_sft_data)} 条 → {consolidated_sft}")
        if all_dpo_data:
            consolidated_dpo = out_dir / "all_dpo_data.json"
            consolidated_dpo.write_text(
                json.dumps(all_dpo_data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            log(f"  📦 汇总 DPO 数据: {len(all_dpo_data)} 条 → {consolidated_dpo}")

        # ═══════ 保存报告 ═══════
        report = {
            "forgex_type": "exam_learning", "mode": config.exam_mode,
            "rounds": round_scores, "total_sft": len(all_sft_data),
            "total_dpo": len(all_dpo_data), "teacher": config.teacher_model_name,
            "student": config.student_model, "final_model": final_model_path,
            "enhancements": {
                "cot": config.cot_distill, "best_of_n": config.best_of_n,
                "progressive_rank": config.progressive_rank,
                "topic_expert": config.topic_expert,
            },
            "output_files": {
                "final_model": final_model_path,
                "sft_data": str(out_dir / "all_sft_data.json") if all_sft_data else None,
                "dpo_data": str(out_dir / "all_dpo_data.json") if all_dpo_data else None,
            },
        }
        (out_dir / "exam_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

        trend = " → ".join([f"R{r['round']}:{r['avg']:.1f}" for r in round_scores])
        progress_cb(98, f"📊 成绩: {trend}")
        summary = (f"✅ 完成 {total_rounds}轮 | SFT {len(all_sft_data)} DPO {len(all_dpo_data)}"
                   f" | 最终模型: {Path(final_model_path).name}")
        progress_cb(100, summary)
        return f"完成 | {trend} | SFT {len(all_sft_data)} DPO {len(all_dpo_data)} | 最终: {final_model_path}"

    # ---- 学生模型: 加载/答题/卸载 ----

    def _resolve_base_model(self, model_path: str) -> Tuple[str, Optional[str]]:
        """解析模型路径，区分完整模型和 LoRA 适配器

        Returns: (base_model_path, lora_adapter_path_or_None)

        自动修复:
        - adapter_config.json 中记录的路径过时 → 搜索 HF cache / loras 目录
        - 完整模型目录直接返回
        """
        p = Path(model_path)
        if not p.is_dir():
            return model_path, None  # HF hub ID，直接用

        adapter_cfg = p / "adapter_config.json"
        has_model_config = (p / "config.json").exists()

        if adapter_cfg.exists() and not has_model_config:
            # 这是 LoRA 适配器目录
            try:
                acfg = json.loads(adapter_cfg.read_text(encoding="utf-8"))
                real_base = acfg.get("base_model_name_or_path", "")
                if real_base:
                    # ═══ 验证路径是否真实可用 ═══
                    resolved = _repair_base_model_path(real_base)
                    if resolved:
                        if resolved != real_base:
                            log(f"  🔧 base_model 路径已修复: {Path(real_base).name} → {Path(resolved).name}")
                        return resolved, str(p)
                    else:
                        # 路径无法修复，给出详细错误
                        raise ValueError(
                            f"LoRA 适配器 {p.name} 的基座模型路径无效:\n"
                            f"  记录路径: {real_base}\n"
                            f"  该路径不存在且无法在 HF 缓存中找到。\n"
                            f"  解决方案:\n"
                            f"    1. 重新下载基座模型: 确保 {real_base} 可访问\n"
                            f"    2. 手动修改 {adapter_cfg} 中的 base_model_name_or_path\n"
                            f"    3. 使用完整模型路径（而非 LoRA 适配器）"
                        )
            except ValueError:
                raise  # 重新抛出上面的 ValueError
            except Exception:
                pass
            return model_path, None  # fallback
        return model_path, None  # 完整模型

    def _load_student_model(self, model_path: str):
        """加载学生模型（支持完整模型和 LoRA 适配器路径）

        加载策略（多级降级）:
        1. fp16/bf16 + device_map="auto"
        2. int8 量化（如果 VRAM 不足）
        3. CPU 加载（最后手段）

        如果 model_path 是 LoRA 适配器目录:
        1. 从 adapter_config.json 读取真正的 base model
        2. 加载 base model
        3. 加载 LoRA adapter
        4. merge_and_unload() 合并为单一模型
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            base_path, lora_path = self._resolve_base_model(model_path)

            # ═══ VRAM 预检查 ═══
            vram = _vram_status()
            est_mb = _estimate_model_vram_mb(base_path if Path(base_path).is_dir() else model_path)
            if vram["total"] > 0 and est_mb > 0:
                log(f"  📊 模型预估 {est_mb:.0f}MB | GPU 剩余 {vram['free']:.0f}MB / {vram['total']:.0f}MB")
                if est_mb > vram["free"] * 0.85:
                    log(f"  ⚠️ 显存可能不足，将尝试量化或 CPU 加载")

            # Tokenizer（从原始路径加载，LoRA 目录里也有 tokenizer）
            tok_path = model_path if Path(model_path).is_dir() else base_path
            tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # ═══ 多级加载策略 ═══
            model = None
            load_method = ""

            # Level 1: 标准 fp16/bf16 + GPU
            if torch.cuda.is_available():
                dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        base_path, torch_dtype=dtype, device_map="auto",
                        trust_remote_code=True, low_cpu_mem_usage=True,
                    )
                    load_method = f"GPU ({dtype})"
                except Exception as e1:
                    err_msg = str(e1).lower()
                    log(f"  ⚠️ GPU 加载失败: {e1}")

                    # Level 2: int8 量化
                    if "out of memory" in err_msg or "oom" in err_msg or "cuda" in err_msg:
                        torch.cuda.empty_cache()
                        try:
                            from transformers import BitsAndBytesConfig
                            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
                            model = AutoModelForCausalLM.from_pretrained(
                                base_path, quantization_config=bnb_cfg,
                                device_map="auto", trust_remote_code=True,
                                low_cpu_mem_usage=True,
                            )
                            load_method = "GPU (int8)"
                            log(f"  🔧 自动降级为 int8 量化")
                        except Exception as e2:
                            log(f"  ⚠️ int8 也失败: {e2}")

            # Level 3: CPU 加载（最后手段）
            if model is None:
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        base_path, torch_dtype=torch.float32,
                        trust_remote_code=True, low_cpu_mem_usage=True,
                    )
                    load_method = "CPU (fp32)"
                    log(f"  📌 使用 CPU 加载（推理速度会较慢）")
                except Exception as e3:
                    log(f"  ❌ 所有加载方式均失败: {e3}")
                    return None, None

            # 如果有 LoRA adapter，加载并合并
            if lora_path and model is not None:
                try:
                    from peft import PeftModel
                    log(f"  📎 加载 LoRA 适配器: {lora_path}")
                    model = PeftModel.from_pretrained(model, lora_path)
                    model = model.merge_and_unload()
                    log(f"  ✅ LoRA 已合并到基座模型")
                except ImportError:
                    log(f"  ⚠️ peft 未安装，无法加载 LoRA。使用基座模型。")
                except Exception as e:
                    log(f"  ⚠️ LoRA 加载失败({e})，使用基座模型。")

            model.eval()
            _vram_log(f"  ✅ 学生模型加载完成 [{load_method}]:")
            return model, tokenizer
        except Exception as e:
            log(f"  ❌ 学生模型加载失败: {e}")
            return None, None

    def _merge_lora_for_chain(self, lora_dir: str, round_label: str, task=None) -> Optional[str]:
        """链式训练: 将 LoRA 合并到基座模型，保存完整模型供下一轮使用

        Returns: 合并后的模型路径, 或 None（失败时）
        """
        progress_cb = task.update_progress if task else lambda p, m="": log(m)
        lora_p = Path(lora_dir)
        adapter_cfg_file = lora_p / "adapter_config.json"

        if not adapter_cfg_file.exists():
            log(f"  {round_label}: 不是 LoRA 目录，跳过合并")
            return lora_dir  # 可能已经是完整模型

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            # ═══ 关键: 合并前先释放所有 GPU 显存 ═══
            # 上一步训练可能残留未释放的 CUDA 张量
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            acfg = json.loads(adapter_cfg_file.read_text(encoding="utf-8"))
            base_model = acfg.get("base_model_name_or_path", "")
            if not base_model:
                log(f"  {round_label}: adapter_config 缺少 base_model_name_or_path")
                return None

            # ═══ 修复过时路径 ═══
            resolved = _repair_base_model_path(base_model)
            if not resolved:
                log(f"  {round_label}: 基座模型路径无效: {base_model}")
                log(f"    提示: 确保基座模型已下载或路径正确")
                return None
            if resolved != base_model:
                log(f"  🔧 {round_label}: 路径修复 → {Path(resolved).name}")
            base_model = resolved

            log(f"  🔀 {round_label}: 合并 LoRA → {Path(base_model).name}")
            _vram_log(f"  合并前")

            dtype = torch.float16
            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability()
                if cap[0] >= 8:
                    dtype = torch.bfloat16

            # 尝试 GPU 加载，OOM 时自动降级到 CPU
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    base_model, torch_dtype=dtype, device_map="auto",
                    trust_remote_code=True, low_cpu_mem_usage=True,
                )
            except (RuntimeError, torch.cuda.OutOfMemoryError) as oom:
                log(f"  ⚠️ {round_label}: GPU 加载 OOM，切换 CPU 合并: {oom}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                model = AutoModelForCausalLM.from_pretrained(
                    base_model, torch_dtype=dtype,
                    trust_remote_code=True, low_cpu_mem_usage=True,
                    device_map="cpu",
                )

            # 优先用训练输出的 tokenizer（可能包含新增 special tokens）
            try:
                tokenizer = AutoTokenizer.from_pretrained(lora_dir, trust_remote_code=True)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

            model = PeftModel.from_pretrained(model, lora_dir)
            model = model.merge_and_unload()

            # 保存到 LoRA 目录旁边的 _merged 目录
            merged_dir = lora_p.parent / f"{lora_p.name}_merged"
            merged_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(merged_dir), safe_serialization=True)
            tokenizer.save_pretrained(str(merged_dir))

            # 清理显存
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            _vram_log(f"  合并后")
            log(f"  ✅ {round_label}: 合并完成 → {merged_dir}")
            return str(merged_dir)

        except ImportError:
            log(f"  ⚠️ {round_label}: peft 未安装，无法合并 LoRA")
            return None
        except Exception as e:
            log(f"  ⚠️ {round_label}: 合并失败({e})")
            return None

    def _force_gc(self):
        """强制释放 GPU 显存 — 在 del 引用后调用"""
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _student_answer_one(self, model, tokenizer, instruction: str, input_text: str = "") -> str:
        """学生回答单个问题"""
        if model is None or tokenizer is None:
            return ""
        try:
            import torch
            full_text = f"{instruction}\n{input_text}".strip() if input_text else instruction

            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                try:
                    chat_input = tokenizer.apply_chat_template(
                        [{"role": "user", "content": full_text}],
                        tokenize=False, add_generation_prompt=True,
                    )
                except Exception:
                    chat_input = full_text
            else:
                chat_input = full_text

            toks = tokenizer(chat_input, return_tensors="pt", truncation=True, max_length=1024)
            input_ids = toks["input_ids"]
            attn_mask = toks.get("attention_mask")

            # ═══ 关键: 检测模型实际所在设备，而非盲目使用 CUDA ═══
            # 模型可能在 CPU 上（如 VRAM 不足时自动降级加载），
            # 此时 torch.cuda.is_available()=True 但模型在 CPU → 设备不匹配崩溃
            try:
                model_device = next(model.parameters()).device
            except StopIteration:
                model_device = torch.device("cpu")

            input_ids = input_ids.to(model_device)
            if attn_mask is not None:
                attn_mask = attn_mask.to(model_device)

            gen_kw = dict(max_new_tokens=512, do_sample=True, temperature=0.7,
                          pad_token_id=tokenizer.eos_token_id, use_cache=False)
            if attn_mask is not None:
                gen_kw["attention_mask"] = attn_mask

            with torch.no_grad():
                out = model.generate(input_ids, **gen_kw)
            return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:
            log(f"    答题失败: {e}")
            return ""

    def _get_teacher_answer(self, client, config: APITeacherConfig,
                             instruction: str, input_text: str) -> str:
        """阶段1: 教师独立回答题目（用于 SFT 数据 + BoN 评分基准）

        CoT 模式下教师输出完整推理过程 + 最终答案。
        """
        if config.cot_distill:
            answer_prompt = (
                "请一步步思考并详细回答以下问题。\n"
                "格式要求:\n"
                "1. 先分析问题的关键点\n"
                "2. 逐步推理得出结论\n"
                "3. 给出完整的最终答案\n\n"
                f"{instruction}"
                + (f"\n\n补充信息: {input_text}" if input_text else "")
            )
        else:
            answer_prompt = instruction + (f"\n\n{input_text}" if input_text else "")
        reply = self._call_api(client, config, instruction=answer_prompt, input_text="")
        return reply.strip()

    def _grade_against_answer(self, client, config: APITeacherConfig,
                               instruction: str, input_text: str,
                               teacher_answer: str, student_answer: str) -> int:
        """阶段2: 对比教师标准答案 vs 学生答案，返回 1-5 分

        BoN 模式下: 教师只答一次，此方法被调用 N 次来评判 N 个候选。
        """
        if not student_answer:
            return 1
        grade_prompt = (
            "你是严格的考试评审。对比标准答案和学生答案，给出评分。\n\n"
            f"【题目】{instruction}\n"
            + (f"【附加信息】{input_text}\n" if input_text else "")
            + f"【标准答案】{teacher_answer[:600]}\n"
            f"【学生答案】{student_answer[:600]}\n\n"
            "评分标准:\n"
            "  5分: 完全正确，推理清晰，与标准答案一致或更好\n"
            "  4分: 基本正确，核心正确但表述/细节有小瑕疵\n"
            "  3分: 部分正确，理解了概念但有明显遗漏或错误\n"
            "  2分: 大部分错误，仅少量内容相关\n"
            "  1分: 完全错误、偏题、或无实质内容\n\n"
            '仅回复JSON: {"score": 1-5整数, "weakness": "主要问题一句话"}'
        )
        try:
            reply = self._call_api(client, config, instruction=grade_prompt, input_text="")
            m = re.search(r'\{[^{}]+\}', reply, re.DOTALL)
            if m:
                parsed = json.loads(m.group())
                return max(1, min(5, int(parsed.get("score", 3))))
            nums = re.findall(r'[1-5]', reply[:50])
            return int(nums[0]) if nums else 3
        except Exception:
            return 3

    def _teacher_grade_one(self, client, config: APITeacherConfig,
                            instruction: str, input_text: str, student_answer: str) -> Dict:
        """教师批改单个题目（两阶段: ①教师先答 ②对比评分）

        单答案模式（非 BoN）使用此方法。
        BoN 模式直接调用 _get_teacher_answer + _grade_against_answer 以复用教师答案。
        """
        result = {"instruction": instruction, "input": input_text, "student_answer": student_answer or ""}

        # ═══ 阶段1: 教师独立回答 ═══
        try:
            teacher_answer = self._get_teacher_answer(client, config, instruction, input_text)
            result["teacher_answer"] = teacher_answer
        except Exception as e:
            result["teacher_answer"] = ""
            result["score"] = 1
            result["weakness"] = f"API错误: {e}"
            return result

        if not student_answer:
            result["score"] = 1
            result["weakness"] = "未作答"
            return result

        # ═══ 阶段2: 对比评分 ═══
        grade_prompt = (
            "你是严格的考试评审。对比标准答案和学生答案，给出评分。\n\n"
            f"【题目】{instruction}\n"
            + (f"【附加信息】{input_text}\n" if input_text else "")
            + f"【标准答案】{teacher_answer[:600]}\n"
            f"【学生答案】{student_answer[:600]}\n\n"
            "评分标准:\n"
            "  5分: 完全正确，推理清晰，与标准答案一致或更好\n"
            "  4分: 基本正确，核心正确但表述/细节有小瑕疵\n"
            "  3分: 部分正确，理解了概念但有明显遗漏或错误\n"
            "  2分: 大部分错误，仅少量内容相关\n"
            "  1分: 完全错误、偏题、或无实质内容\n\n"
            '仅回复JSON: {"score": 1-5整数, "weakness": "主要问题一句话"}'
        )
        try:
            reply = self._call_api(client, config, instruction=grade_prompt, input_text="")
            m = re.search(r'\{[^{}]+\}', reply, re.DOTALL)
            if m:
                parsed = json.loads(m.group())
                result["score"] = max(1, min(5, int(parsed.get("score", 3))))
                result["weakness"] = parsed.get("weakness", "")
            else:
                nums = re.findall(r'[1-5]', reply[:50])
                result["score"] = int(nums[0]) if nums else 3
                result["weakness"] = reply[:100]
        except (json.JSONDecodeError, ValueError, IndexError):
            result["score"] = 3
            result["weakness"] = ""
        except Exception:
            result["score"] = 3
            result["weakness"] = ""

        return result

    def _train_topic_experts(self, client, config: APITeacherConfig,
                              sft_data: List[Dict], out_dir: Path,
                              base_model: str, task, trainer_engine) -> List[str]:
        """主题专家 MoLoRA: 按主题训练专门的 LoRA 适配器

        原理（类似 Mixture-of-Experts 的 LoRA 版本）:
        1. API 教师将所有训练数据分类到 K 个主题
        2. 每个主题单独训练一个专精 LoRA（低 rank，专注特定能力）
        3. 每个专家 LoRA 合并为独立可用的完整模型
        4. 保存路由信息 + 关键词表 → 推理时按主题匹配加载对应模型

        这让小模型用有限参数覆盖更广的知识面:
        - 通用 LoRA: 1×64rank = 64 的表达力
        - 4个专家: 4×32rank = 128 的总表达力
        """
        k = max(2, min(8, config.topic_expert_k))
        progress_cb = task.update_progress if task else lambda p, m="": log(m)

        # ═══ Step 1: API 教师分类 ═══
        import random
        samples = random.sample(sft_data, min(20, len(sft_data)))
        sample_text = "\n".join([f"- {s['instruction'][:80]}" for s in samples])

        classify_prompt = (
            f"根据以下训练数据样本，将它们归纳为恰好 {k} 个主题类别。\n"
            f"每个类别给出: 简短标签（2-6字）和5个代表性关键词。\n\n"
            f"样本:\n{sample_text}\n\n"
            f'仅回复JSON数组: [{{"label": "主题名", "keywords": ["词1","词2","词3","词4","词5"]}}, ...]'
        )

        topics = []
        topic_keywords: Dict[str, List[str]] = {}
        try:
            reply = self._call_api(client, config, instruction=classify_prompt, input_text="")
            # 尝试解析带关键词的格式
            m = re.search(r'\[.*\]', reply, re.DOTALL)
            if m:
                parsed = json.loads(m.group())
                for item in parsed:
                    if isinstance(item, dict):
                        label = str(item.get("label", "")).strip()
                        kws = item.get("keywords", [])
                        if label:
                            topics.append(label)
                            topic_keywords[label] = [str(w) for w in kws][:10]
                    elif isinstance(item, str):
                        topics.append(item.strip())
                topics = topics[:k]
        except Exception:
            pass

        if len(topics) < 2:
            topics = [f"主题{i+1}" for i in range(k)]
            topic_keywords = {}

        log(f"  📂 主题: {topics}")
        if topic_keywords:
            for t, kws in topic_keywords.items():
                log(f"    {t}: {', '.join(kws)}")

        # ═══ Step 2: 将数据分配到各主题 ═══
        topic_buckets: Dict[str, List[Dict]] = {t: [] for t in topics}
        batch_size = 20

        for bi in range(0, len(sft_data), batch_size):
            batch = sft_data[bi:bi + batch_size]
            batch_text = "\n".join([
                f"{j}. {item['instruction'][:60]}"
                for j, item in enumerate(batch)
            ])
            topics_str = ", ".join([f'"{t}"' for t in topics])

            assign_prompt = (
                f"将以下问题分配到最匹配的主题。主题列表: [{topics_str}]\n\n"
                f"{batch_text}\n\n"
                f'回复JSON: {{"0": "主题名", "1": "主题名", ...}}'
            )

            try:
                reply = self._call_api(client, config, instruction=assign_prompt, input_text="")
                m = re.search(r'\{[^{}]*\}', reply, re.DOTALL)
                if m:
                    mapping = json.loads(m.group())
                    for j, item in enumerate(batch):
                        assigned = mapping.get(str(j), topics[0])
                        best_topic = topics[0]
                        for t in topics:
                            if t in str(assigned) or str(assigned) in t:
                                best_topic = t
                                break
                        topic_buckets[best_topic].append(item)
                else:
                    for j, item in enumerate(batch):
                        topic_buckets[topics[j % len(topics)]].append(item)
            except Exception:
                for j, item in enumerate(batch):
                    topic_buckets[topics[j % len(topics)]].append(item)

        # 过滤太少数据的主题
        min_samples = 5
        active_topics = {t: items for t, items in topic_buckets.items() if len(items) >= min_samples}
        if not active_topics:
            log("  每个主题数据太少，跳过专家训练")
            return []

        for t, items in active_topics.items():
            log(f"    {t}: {len(items)} 条")

        # ═══ Step 3: 训练各主题专家 LoRA → 合并为完整模型 ═══
        expert_rank = max(16, config.rank // 2)
        expert_paths = []
        expert_info = []  # 用于路由

        for ti, (topic, items) in enumerate(active_topics.items()):
            safe_name = re.sub(r'[^\w]', '_', topic)[:20]
            expert_name = f"{config.output_name}_expert_{safe_name}"
            expert_file = out_dir / f"expert_{safe_name}.json"
            expert_file.write_text(json.dumps(items, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

            progress_cb(91 + ti * 6 // max(len(active_topics), 1),
                        f"🧠 专家 [{topic}]: {len(items)}条 rank={expert_rank}")
            try:
                trainer_engine.train(
                    method="sft", backend="auto",
                    base_model=base_model,
                    dataset_path=str(expert_file),
                    params={
                        "output_name": expert_name,
                        "lr": config.lr * 0.8,
                        "batch_size": config.batch_size,
                        "epochs": max(1.0, config.epochs * 0.5),
                        "max_seq_len": config.max_seq_len,
                        "rank": expert_rank,
                        "use_qlora": config.use_qlora,
                        "gradient_accumulation_steps": config.gradient_accumulation_steps,
                    },
                    task=task,
                )
                expert_lora = Path("data/loras") / expert_name
                if expert_lora.exists():
                    # 合并专家 LoRA → 完整可用模型
                    merged = self._merge_lora_for_chain(
                        str(expert_lora), f"专家[{topic}]", task)
                    final_path = merged or str(expert_lora)
                    expert_paths.append(final_path)
                    expert_info.append({
                        "topic": topic,
                        "keywords": topic_keywords.get(topic, []),
                        "model_path": final_path,
                        "data_count": len(items),
                        "rank": expert_rank,
                    })
                    log(f"  ✅ 专家 [{topic}]: {final_path}")
            except Exception as e:
                log(f"  ⚠️ 专家 [{topic}] 训练失败: {e}")

        # ═══ Step 4: 保存完整路由信息 ═══
        if expert_info:
            router_info = {
                "type": "molora_experts",
                "base_model": base_model,
                "experts": expert_info,
                "usage": {
                    "说明": "根据用户问题匹配主题关键词，加载对应 model_path 的模型",
                    "匹配方式": "检查问题是否包含 keywords 中的关键词，命中最多的主题即为目标",
                    "默认模型": base_model,
                },
            }
            (out_dir / "expert_router.json").write_text(
                json.dumps(router_info, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

            # 保存各专家的数据集（方便用户单独使用）
            (out_dir / "expert_datasets.json").write_text(
                json.dumps({t: len(items) for t, items in active_topics.items()},
                           indent=2, ensure_ascii=False, default=str), encoding="utf-8")

            log(f"  📄 路由信息已保存: expert_router.json ({len(expert_info)} 个专家)")

        return expert_paths

    def _generate_exam_questions(self, client, config: APITeacherConfig,
                                  seed_prompts: List[Dict], n_questions: int,
                                  round_num: int, task=None,
                                  base_pct: int = 10, pct_range: int = 10) -> List[Dict]:
        """用 API 根据种子数据的主题方向生成全新考试题目"""
        import random
        progress_cb = task.update_progress if task else lambda p, m="": log(m)

        # 从种子数据中抽取样例作为主题参考
        sample_size = min(20, len(seed_prompts))
        samples = random.sample(seed_prompts, sample_size)
        sample_texts = "\n".join([f"- {s.get('instruction', '')[:100]}" for s in samples])

        # 分批生成（每批 10-20 题，避免 API 一次生不出来）
        batch_size = min(15, n_questions)
        all_questions = []
        batches_needed = math.ceil(n_questions / batch_size)

        for bi in range(batches_needed):
            remaining = n_questions - len(all_questions)
            this_batch = min(batch_size, remaining)
            if this_batch <= 0:
                break

            gen_prompt = (
                f"你是一位出题专家。请根据以下主题方向，生成 {this_batch} 道全新的考试题目。\n\n"
                f"【参考主题】:\n{sample_texts}\n\n"
                f"【要求】:\n"
                f"1. 题目要多样化，覆盖不同难度（简单/中等/困难）\n"
                f"2. 这是第 {round_num} 轮考试，请适当提高难度\n"
                f"3. 每道题独占一行，不要编号，不要多余格式\n"
                f'4. 直接输出题目内容，不要加"题目"等前缀\n\n'
                f"请生成 {this_batch} 道题："
            )

            try:
                reply = self._call_api(client, config, instruction=gen_prompt)
                if reply:
                    lines = [l.strip() for l in reply.strip().splitlines() if l.strip()]
                    # 清理: 去掉编号前缀
                    for line in lines:
                        clean = re.sub(r'^[\d]+[.、)\]：:]\s*', '', line).strip()
                        clean = re.sub(r'^题目[：:]\s*', '', clean).strip()
                        if clean and len(clean) >= 5:
                            all_questions.append({"instruction": clean})
            except Exception as e:
                log(f"  出题批次 {bi+1} 失败: {e}")

            if bi < batches_needed - 1:
                pct = base_pct + int(pct_range * (bi + 1) / batches_needed)
                progress_cb(pct, f"📝 出题: {len(all_questions)}/{n_questions}...")

        log(f"  API 出题完成: {len(all_questions)} 道 (目标 {n_questions})")
        return all_questions[:n_questions]

    def _save_exam_chart(self, out_dir: Path, round_scores: List[Dict], config):
        """生成 HTML 成绩可视化图表"""
        if not round_scores:
            return

        rounds_js = json.dumps(round_scores, ensure_ascii=False, default=str)
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>ForgeX 做题学习报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; background: #0d1117; color: #c9d1d9; }}
h1 {{ color: #58a6ff; }} h2 {{ color: #79c0ff; border-bottom: 1px solid #21262d; padding-bottom: 8px; }}
.card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin: 16px 0; }}
.stats {{ display: flex; gap: 20px; flex-wrap: wrap; }}
.stat {{ text-align: center; flex: 1; min-width: 100px; }}
.stat .num {{ font-size: 2em; font-weight: bold; color: #58a6ff; }}
.stat .label {{ color: #8b949e; font-size: 0.9em; }}
canvas {{ max-height: 350px; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 8px 12px; text-align: center; border-bottom: 1px solid #21262d; }}
th {{ color: #58a6ff; }}
.score-bar {{ display: inline-block; height: 14px; border-radius: 3px; }}
</style></head><body>
<h1>📊 ForgeX 做题学习报告</h1>
<p>教师: {config.teacher_model_name} → 学生: {config.student_model}</p>

<div class="card"><div class="stats" id="summary"></div></div>

<div class="card">
<h2>📈 成绩趋势</h2>
<canvas id="trendChart"></canvas>
</div>

<div class="card">
<h2>📊 各轮详情</h2>
<table><thead><tr><th>轮次</th><th>均分</th><th>✅正确</th><th>❌错误</th><th>🔇未答</th><th>总题</th><th>分数分布</th></tr></thead>
<tbody id="detailTable"></tbody></table>
</div>

<div class="card">
<h2>🎯 分数分布变化</h2>
<canvas id="distChart"></canvas>
</div>

<script>
const data = {rounds_js};
const last = data[data.length - 1];
const first = data[0];

// Summary
document.getElementById('summary').innerHTML = `
  <div class="stat"><div class="num">${{data.length}}</div><div class="label">总轮数</div></div>
  <div class="stat"><div class="num">${{first.avg.toFixed(1)}}</div><div class="label">初始均分</div></div>
  <div class="stat"><div class="num">${{last.avg.toFixed(1)}}</div><div class="label">最终均分</div></div>
  <div class="stat"><div class="num">${{(last.avg - first.avg > 0 ? '+' : '') + (last.avg - first.avg).toFixed(1)}}</div><div class="label">提升</div></div>
`;

// Detail table
const tbody = document.getElementById('detailTable');
data.forEach(r => {{
  const dist = r.distribution || {{}};
  const maxCount = Math.max(...Object.values(dist), 1);
  let bars = '';
  const colors = ['#f85149','#f0883e','#d29922','#3fb950','#58a6ff'];
  for (let s = 1; s <= 5; s++) {{
    const count = dist[s] || 0;
    const w = Math.max(2, count / maxCount * 80);
    bars += `<span class="score-bar" style="width:${{w}}px;background:${{colors[s-1]}}" title="${{s}}分: ${{count}}"></span> `;
  }}
  tbody.innerHTML += `<tr><td>R${{r.round}}</td><td>${{r.avg.toFixed(1)}}</td><td>${{r.correct}}</td><td>${{r.wrong}}</td><td>${{r.unanswered||0}}</td><td>${{r.total}}</td><td>${{bars}}</td></tr>`;
}});

// Trend chart
new Chart(document.getElementById('trendChart'), {{
  type: 'line',
  data: {{
    labels: data.map(r => 'R' + r.round),
    datasets: [
      {{ label: '均分', data: data.map(r => r.avg), borderColor: '#58a6ff', backgroundColor: 'rgba(88,166,255,0.1)', fill: true, tension: 0.3 }},
      {{ label: '正确率%', data: data.map(r => r.total ? Math.round(r.correct/r.total*100) : 0), borderColor: '#3fb950', borderDash: [5,5], yAxisID: 'pct' }}
    ]
  }},
  options: {{
    scales: {{
      y: {{ min: 0, max: 5, title: {{ display: true, text: '均分' }}, grid: {{ color: '#21262d' }} }},
      pct: {{ position: 'right', min: 0, max: 100, title: {{ display: true, text: '正确率%' }}, grid: {{ drawOnChartArea: false }} }},
      x: {{ grid: {{ color: '#21262d' }} }}
    }},
    plugins: {{ legend: {{ labels: {{ color: '#c9d1d9' }} }} }}
  }}
}});

// Distribution chart
new Chart(document.getElementById('distChart'), {{
  type: 'bar',
  data: {{
    labels: data.map(r => 'R' + r.round),
    datasets: [1,2,3,4,5].map((s, i) => ({{
      label: s + '分',
      data: data.map(r => (r.distribution || {{}})[s] || 0),
      backgroundColor: ['#f85149','#f0883e','#d29922','#3fb950','#58a6ff'][i]
    }}))
  }},
  options: {{
    scales: {{
      x: {{ stacked: true, grid: {{ color: '#21262d' }} }},
      y: {{ stacked: true, grid: {{ color: '#21262d' }}, title: {{ display: true, text: '题数' }} }}
    }},
    plugins: {{ legend: {{ labels: {{ color: '#c9d1d9' }} }} }}
  }}
}});
</script></body></html>"""

        chart_file = out_dir / "exam_chart.html"
        chart_file.write_text(html, encoding="utf-8")
        log(f"  📊 成绩图表已保存: {chart_file}")


    def _load_prompts(self, config: APITeacherConfig) -> List[Dict]:
        """从数据集加载 instruction prompts"""
        ds_path = Path(config.dataset_path)
        if not ds_path.is_absolute():
            ds_path = Path(DATASETS_DIR) / config.dataset_path
        if not ds_path.exists():
            raise FileNotFoundError(f"种子数据不存在: {ds_path}")

        # 如果是目录 → 查找其中的 JSON/JSONL 文件
        if ds_path.is_dir():
            candidates = (
                list(ds_path.glob("*.jsonl")) +
                list(ds_path.glob("*.json")) +
                list(ds_path.glob("*.txt"))
            )
            if not candidates:
                raise FileNotFoundError(
                    f"数据集目录 {ds_path} 中没有找到 .json/.jsonl/.txt 文件。\n"
                    f"请在目录中放置包含 instruction 字段的 JSON/JSONL 文件。"
                )
            # 优先选 jsonl > json > txt，选最大的文件
            candidates.sort(key=lambda f: (
                0 if f.suffix == '.jsonl' else 1 if f.suffix == '.json' else 2,
                -f.stat().st_size
            ))
            ds_path = candidates[0]
            log(f"📂 数据集目录，自动选择: {ds_path.name}")

        text = ds_path.read_text(encoding="utf-8")
        rows = []
        if ds_path.suffix == ".jsonl":
            for line in text.strip().splitlines():
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        elif ds_path.suffix == ".json":
            data = json.loads(text)
            rows = data if isinstance(data, list) else [data]
        else:
            # 纯文本 → 每行一个 instruction
            for line in text.strip().splitlines():
                if line.strip():
                    rows.append({"instruction": line.strip()})

        # 提取 instruction
        prompts = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            if "instruction" in r:
                prompts.append(r)
            elif "messages" in r or "conversations" in r:
                msgs = r.get("messages") or r.get("conversations") or []
                user_msgs = [m["content"] for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
                if user_msgs:
                    prompts.append({"instruction": user_msgs[0]})
            elif "text" in r:
                prompts.append({"instruction": str(r["text"])[:500]})
            elif "prompt" in r:
                prompts.append({"instruction": str(r["prompt"])[:500]})
            elif "question" in r:
                prompts.append({"instruction": str(r["question"])[:500]})
        return prompts

    def _make_client(self, config: APITeacherConfig):
        """创建 API 客户端（支持 OpenAI / DeepSeek / Anthropic Claude）"""
        import urllib.request
        import urllib.error
        import ssl

        class SimpleClient:
            def __init__(self, base, key, model):
                self.base = base.rstrip("/")
                self.key = key
                self.model = model
                self._ssl_ctx = None
                # 自动检测 API 类型
                self._api_type = self._detect_api_type()

            def _detect_api_type(self):
                """根据 base URL 和模型名自动检测 API 类型"""
                base_lower = self.base.lower()
                model_lower = self.model.lower()
                if "anthropic" in base_lower or model_lower.startswith("claude"):
                    return "anthropic"
                return "openai"  # OpenAI / DeepSeek / vLLM 都兼容

            def _get_ssl_context(self):
                """获取 SSL 上下文（延迟创建，不做预测试）"""
                if self._ssl_ctx is not None:
                    return self._ssl_ctx
                self._ssl_ctx = ssl.create_default_context()
                return self._ssl_ctx

            def _try_request(self, url, body_bytes, headers):
                """发送请求，带 SSL 自动降级和重试"""
                last_err = None
                ssl_verified = True

                for attempt in range(3):
                    try:
                        req = urllib.request.Request(url, data=body_bytes, headers=headers)
                        ctx = self._get_ssl_context()
                        kwargs = {"timeout": 60}
                        if ctx:
                            kwargs["context"] = ctx
                        with urllib.request.urlopen(req, **kwargs) as resp:
                            return json.loads(resp.read().decode("utf-8"))

                    except urllib.error.HTTPError as e:
                        err_body = ""
                        try:
                            err_body = e.read().decode("utf-8", errors="replace")[:500]
                        except Exception:
                            pass
                        status = e.code
                        if status == 401:
                            raise RuntimeError(f"API 认证失败 (401)。请检查 API Key 是否正确。\n{err_body}") from e
                        elif status == 403:
                            raise RuntimeError(f"API 权限不足 (403)。请检查 API Key 权限或账户余额。\n{err_body}") from e
                        elif status == 404:
                            raise RuntimeError(
                                f"API 端点不存在 (404)。请检查:\n"
                                f"  - URL: {url}\n"
                                f"  - 模型: {self.model}\n"
                                f"  - 如用 Claude，API 端点填: https://api.anthropic.com\n"
                                f"  - 如用 DeepSeek，API 端点填: https://api.deepseek.com\n"
                                f"详情: {err_body}"
                            ) from e
                        elif status == 429:
                            import time as _time
                            wait = min(2 ** attempt * 3, 30)
                            log(f"  API 速率限制 (429)，等待 {wait}s 后重试...")
                            _time.sleep(wait)
                            last_err = e
                            continue
                        elif status >= 500:
                            import time as _time
                            _time.sleep(2 ** attempt)
                            last_err = e
                            continue
                        else:
                            raise RuntimeError(f"API 请求失败 ({status}): {err_body}") from e

                    except urllib.error.URLError as e:
                        reason = str(e.reason) if hasattr(e, "reason") else str(e)
                        if ssl_verified and ("SSL" in reason or "CERTIFICATE" in reason.upper()):
                            log("⚠️ SSL 证书错误，切换到不验证模式...")
                            ctx = ssl.create_default_context()
                            ctx.check_hostname = False
                            ctx.verify_mode = ssl.CERT_NONE
                            self._ssl_ctx = ctx
                            ssl_verified = False
                            last_err = e
                            continue
                        raise RuntimeError(
                            f"无法连接 API: {reason}\nURL: {url}\n"
                            f"请检查: 1) API 端点是否正确 2) 网络是否通畅 3) 是否需要代理"
                        ) from e

                    except (TimeoutError, ConnectionError, OSError) as e:
                        import time as _time
                        _time.sleep(2 ** attempt)
                        last_err = e
                        continue

                raise RuntimeError(f"API 调用失败（已重试 3 次）: {last_err}")

            def chat(self, messages, temperature=0.7, max_tokens=2048, top_p=0.95):
                if self._api_type == "anthropic":
                    return self._chat_anthropic(messages, temperature, max_tokens, top_p)
                return self._chat_openai(messages, temperature, max_tokens, top_p)

            def _chat_openai(self, messages, temperature=0.7, max_tokens=2048, top_p=0.95):
                """OpenAI / DeepSeek / vLLM 兼容格式"""
                url = f"{self.base}/chat/completions"
                headers = {"Content-Type": "application/json"}
                if self.key:
                    headers["Authorization"] = f"Bearer {self.key}"
                body = json.dumps({
                    "model": self.model, "messages": messages,
                    "temperature": temperature, "max_tokens": max_tokens,
                    "top_p": top_p,
                }, default=str).encode("utf-8")

                data = self._try_request(url, body, headers)
                if "error" in data:
                    err = data["error"]
                    msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                    raise RuntimeError(f"API 错误: {msg}")
                choices = data.get("choices", [])
                if not choices:
                    raise RuntimeError(f"API 返回空结果: {json.dumps(data, ensure_ascii=False, default=str)[:200]}")
                return choices[0].get("message", {}).get("content", "") or choices[0].get("text", "")

            def _chat_anthropic(self, messages, temperature=0.7, max_tokens=2048, top_p=0.95):
                """Anthropic Claude API 原生格式"""
                base = self.base.rstrip("/")
                if not base.endswith("/v1"):
                    base = base + "/v1"
                url = f"{base}/messages"

                headers = {
                    "Content-Type": "application/json",
                    "x-api-key": self.key,
                    "anthropic-version": "2023-06-01",
                }
                system_text = ""
                api_messages = []
                for m in messages:
                    if m.get("role") == "system":
                        system_text = m.get("content", "")
                    else:
                        api_messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})
                if not api_messages:
                    api_messages = [{"role": "user", "content": "Hello"}]

                payload = {
                    "model": self.model, "messages": api_messages,
                    "max_tokens": max_tokens, "temperature": temperature,
                    "top_p": top_p,
                }
                if system_text:
                    payload["system"] = system_text
                body = json.dumps(payload, default=str).encode("utf-8")
                data = self._try_request(url, body, headers)

                if "error" in data:
                    err = data["error"]
                    msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                    raise RuntimeError(f"Claude API 错误: {msg}")
                content_blocks = data.get("content", [])
                if not content_blocks:
                    raise RuntimeError(f"Claude API 空结果: {json.dumps(data, ensure_ascii=False, default=str)[:200]}")
                return "\n".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")

        return SimpleClient(config.api_base, config.api_key, config.teacher_model_name)

    def _call_api(self, client, config: APITeacherConfig, instruction: str, input_text: str = "") -> str:
        """调用 API 生成教师回复"""
        messages = []
        if config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})

        user_msg = instruction
        if input_text:
            user_msg += f"\n\n{input_text}"
        messages.append({"role": "user", "content": user_msg})

        return client.chat(
            messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
        )


# 单例
api_teacher = APITeacherDistiller()
