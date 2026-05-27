# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

"""ForgeX 训练模拟器 — 训练前预估 VRAM、时间、最优参数。

核心功能：
  1. VRAM 精确预估（模型权重 + LoRA + 优化器 + 激活值 + 梯度）
  2. 训练时间预估（基于 GPU tokens/sec 基准数据）
  3. 智能参数推荐（给定 VRAM 预算，推荐最优配置）
  4. 训练收益预测（根据数据集大小/质量估算效果）
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, List


# ===================== GPU 吞吐量基准数据 =====================
# tokens_per_sec_per_b: 每 B 参数在该 GPU 上的 tokens/sec（LoRA FP16 训练）
# 基于社区基准数据 + 经验值
GPU_THROUGHPUT = {
    # GPU pattern: (tok/s per B param for FP16 LoRA, tok/s per B for QLoRA)
    "4090":      (180, 120),
    "4080":      (130, 90),
    "4070 ti":   (100, 70),
    "4070":      (85, 60),
    "4060 ti":   (55, 40),
    "4060":      (50, 35),
    "3090":      (140, 95),
    "3080":      (90, 65),
    "3070":      (60, 42),
    "3060":      (55, 38),
    "a100":      (350, 230),
    "h100":      (500, 340),
    "l40s":      (280, 190),
    "l40":       (260, 175),
    "l4":        (120, 80),
    "t4":        (40, 28),
    "2080 ti":   (55, 38),
    "2080":      (45, 32),
    "2070":      (38, 26),
    "2060":      (28, 20),
}


@dataclass
class VRAMBreakdown:
    """VRAM 分项明细"""
    model_weights_mb: float = 0
    lora_weights_mb: float = 0
    optimizer_states_mb: float = 0
    gradients_mb: float = 0
    activations_mb: float = 0
    kv_cache_mb: float = 0
    overhead_mb: float = 0
    total_mb: float = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            "模型权重": round(self.model_weights_mb),
            "LoRA 权重": round(self.lora_weights_mb),
            "优化器状态": round(self.optimizer_states_mb),
            "梯度": round(self.gradients_mb),
            "激活值": round(self.activations_mb),
            "KV Cache": round(self.kv_cache_mb),
            "系统开销": round(self.overhead_mb),
            "总计": round(self.total_mb),
        }

    @property
    def total_gb(self) -> float:
        return self.total_mb / 1024


@dataclass
class TrainingEstimate:
    """训练预估结果"""
    vram: VRAMBreakdown = field(default_factory=VRAMBreakdown)
    time_seconds: float = 0
    time_human: str = ""
    tokens_total: int = 0
    steps_total: int = 0
    feasible: bool = True
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    # 推荐参数
    recommended: Dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = []
        lines.append(f"📊 VRAM 预估: {self.vram.total_gb:.1f} GB")
        lines.append(f"⏱️ 训练时间: {self.time_human}")
        lines.append(f"📝 总 Tokens: {self.tokens_total:,}")
        lines.append(f"🔢 总步数: {self.steps_total:,}")
        if not self.feasible:
            lines.append("⚠️ 当前配置可能不可行！")
        for w in self.warnings:
            lines.append(f"⚠️ {w}")
        for s in self.suggestions:
            lines.append(f"💡 {s}")
        return "\n".join(lines)


class TrainingEstimator:
    """训练参数模拟器"""

    def estimate(
        self,
        model_params_b: float,        # 模型参数量（B）
        dataset_samples: int,          # 数据集样本数
        avg_seq_len: int = 512,        # 平均序列长度
        max_seq_len: int = 2048,       # 最大序列长度
        batch_size: int = 1,           # per-device batch size
        gradient_accumulation: int = 4, # 梯度累积
        epochs: float = 1.0,
        rank: int = 64,                # LoRA rank
        use_qlora: bool = False,
        use_8bit_optim: bool = True,   # paged_adamw_8bit
        gradient_checkpointing: bool = True,
        gpu_name: str = "",
        gpu_vram_mb: int = 0,
    ) -> TrainingEstimate:
        est = TrainingEstimate()
        est.warnings = []
        est.suggestions = []

        # ======== VRAM 计算 ========
        vram = VRAMBreakdown()

        # 1. 模型权重
        if use_qlora:
            # 4bit: ~0.5 bytes/param + 量化开销
            vram.model_weights_mb = model_params_b * 1e9 * 0.55 / (1024 ** 2)
        else:
            # FP16: 2 bytes/param
            vram.model_weights_mb = model_params_b * 1e9 * 2 / (1024 ** 2)

        # 2. LoRA 权重 (FP16)
        # LoRA params ≈ 2 * rank * hidden_dim * num_layers * num_modules
        # 简化估算：全 linear 约占总参数的 0.5-2%
        hidden_dim = int(math.sqrt(model_params_b * 1e9 / 100))  # 粗估 hidden dim
        num_layers = int(model_params_b * 1e9 / (hidden_dim * hidden_dim * 12))  # 粗估层数
        lora_params = 2 * rank * hidden_dim * num_layers * 7  # 7 个 linear modules
        lora_params = min(lora_params, model_params_b * 1e9 * 0.05)  # 上限 5%
        vram.lora_weights_mb = lora_params * 2 / (1024 ** 2)  # FP16

        # 3. 优化器状态
        if use_8bit_optim:
            # paged_adamw_8bit: 1 byte/param for m + 1 byte/param for v
            vram.optimizer_states_mb = lora_params * 2 / (1024 ** 2)
        else:
            # AdamW FP32: 4 bytes/param for m + 4 bytes/param for v
            vram.optimizer_states_mb = lora_params * 8 / (1024 ** 2)

        # 4. 梯度 (FP16)
        vram.gradients_mb = lora_params * 2 / (1024 ** 2)

        # 5. 激活值
        if gradient_checkpointing:
            # ~30% of full activations (only save checkpoint boundaries)
            act_factor = 0.3
        else:
            act_factor = 1.0
        # 激活值 ≈ batch * seq_len * hidden * num_layers * 2 bytes * factor
        act_bytes = batch_size * max_seq_len * hidden_dim * num_layers * 2 * act_factor
        vram.activations_mb = act_bytes / (1024 ** 2)
        # 激活值经验上限（不太可能超过模型本身）
        vram.activations_mb = min(vram.activations_mb, vram.model_weights_mb * 1.5)

        # 6. KV Cache（训练时较小，主要是当前 batch）
        vram.kv_cache_mb = batch_size * max_seq_len * hidden_dim * num_layers * 4 / (1024 ** 2) * 0.1

        # 7. 系统开销（CUDA context + PyTorch allocator + 碎片）
        vram.overhead_mb = max(500, vram.model_weights_mb * 0.08)

        vram.total_mb = (vram.model_weights_mb + vram.lora_weights_mb +
                         vram.optimizer_states_mb + vram.gradients_mb +
                         vram.activations_mb + vram.kv_cache_mb + vram.overhead_mb)
        est.vram = vram

        # ======== 时间计算 ========
        effective_batch = batch_size * gradient_accumulation
        total_tokens = int(dataset_samples * avg_seq_len * epochs)
        steps = int(math.ceil(dataset_samples * epochs / effective_batch))
        est.tokens_total = total_tokens
        est.steps_total = steps

        # 查找 GPU 吞吐量
        tok_per_sec = self._gpu_throughput(gpu_name, model_params_b, use_qlora)
        if tok_per_sec > 0:
            est.time_seconds = total_tokens / tok_per_sec
        else:
            # 无 GPU 基准数据，用保守估计
            est.time_seconds = total_tokens / max(10, model_params_b * 5)

        est.time_human = self._format_time(est.time_seconds)

        # ======== 可行性检查 ========
        if gpu_vram_mb > 0:
            if vram.total_mb > gpu_vram_mb * 0.95:
                est.feasible = False
                est.warnings.append(
                    f"VRAM 不足：需要 {vram.total_gb:.1f}GB，你的 GPU 只有 {gpu_vram_mb/1024:.1f}GB"
                )
                # 给出建议
                if not use_qlora:
                    est.suggestions.append("建议开启 QLoRA（4bit 量化），可减少约 60% 显存")
                if batch_size > 1:
                    est.suggestions.append(f"建议将 batch_size 降到 1（当前 {batch_size}）")
                if rank > 32:
                    est.suggestions.append(f"建议将 LoRA rank 降到 32（当前 {rank}）")
                if max_seq_len > 1024:
                    est.suggestions.append(f"建议将 max_seq_len 降到 1024（当前 {max_seq_len}）")
            elif vram.total_mb > gpu_vram_mb * 0.85:
                est.warnings.append("VRAM 余量较小，训练中可能 OOM。建议降低 batch_size 或开启 QLoRA")
        else:
            est.warnings.append("无法检测 GPU VRAM，无法判断可行性")

        # ======== 训练建议 ========
        if dataset_samples < 100:
            est.warnings.append(f"数据集只有 {dataset_samples} 条，可能不足以有效训练")
            est.suggestions.append("建议至少 500 条以上数据。可以用 AI 数据合成扩充")
        if dataset_samples > 50000 and epochs > 1:
            est.suggestions.append("大数据集通常 1 epoch 即可收敛，建议 epochs=1")
        if avg_seq_len > max_seq_len * 0.8:
            est.warnings.append("平均序列长度接近最大值，部分样本可能被截断")

        # ======== 推荐参数 ========
        if gpu_vram_mb > 0 and not getattr(self, '_in_recommend', False):
            est.recommended = self.recommend_params(
                model_params_b, gpu_vram_mb, dataset_samples, avg_seq_len
            )

        return est

    def recommend_params(
        self,
        model_params_b: float,
        gpu_vram_mb: int,
        dataset_samples: int = 1000,
        avg_seq_len: int = 512,
    ) -> Dict:
        """给定 VRAM 预算，推荐最优训练参数"""
        self._in_recommend = True  # 防止 estimate() 再调回来
        try:
            return self._recommend_params_inner(model_params_b, gpu_vram_mb, dataset_samples, avg_seq_len)
        finally:
            self._in_recommend = False

    def _recommend_params_inner(self, model_params_b, gpu_vram_mb, dataset_samples, avg_seq_len) -> Dict:
        gpu_gb = gpu_vram_mb / 1024

        # 先试 FP16 LoRA（最佳质量）
        configs = []
        for qlora in [False, True]:
            for rank in [128, 64, 32, 16]:
                for bs in [4, 2, 1]:
                    for seq in [4096, 2048, 1024, 512]:
                        est = self.estimate(
                            model_params_b=model_params_b,
                            dataset_samples=dataset_samples,
                            avg_seq_len=avg_seq_len,
                            max_seq_len=seq,
                            batch_size=bs,
                            gradient_accumulation=max(1, 4 // bs),
                            rank=rank,
                            use_qlora=qlora,
                            gpu_vram_mb=gpu_vram_mb,
                        )
                        if est.vram.total_mb < gpu_vram_mb * 0.88:
                            # 打分：更高 rank + 更大 seq + 更大 batch = 更好
                            score = (
                                rank * 2 +            # rank 权重最高
                                seq / 100 +           # 序列长度
                                bs * 10 +             # batch size
                                (0 if qlora else 50)  # FP16 优于 QLoRA
                            )
                            configs.append({
                                "score": score,
                                "use_qlora": qlora,
                                "rank": rank,
                                "batch_size": bs,
                                "gradient_accumulation_steps": max(1, 4 // bs),
                                "max_seq_len": seq,
                                "vram_gb": round(est.vram.total_gb, 1),
                            })

        if not configs:
            return {
                "use_qlora": True,
                "rank": 16,
                "batch_size": 1,
                "gradient_accumulation_steps": 4,
                "max_seq_len": 512,
                "note": "VRAM 极度紧张，建议使用更小的模型",
            }

        best = max(configs, key=lambda c: c["score"])
        best.pop("score")
        return best

    def compare_configs(
        self,
        model_params_b: float,
        dataset_samples: int,
        configs: List[Dict],
        gpu_name: str = "",
        gpu_vram_mb: int = 0,
    ) -> List[Dict]:
        """对比多组配置的预估结果"""
        results = []
        for i, cfg in enumerate(configs):
            est = self.estimate(
                model_params_b=model_params_b,
                dataset_samples=dataset_samples,
                avg_seq_len=cfg.get("avg_seq_len", 512),
                max_seq_len=cfg.get("max_seq_len", 2048),
                batch_size=cfg.get("batch_size", 1),
                gradient_accumulation=cfg.get("gradient_accumulation_steps", 4),
                epochs=cfg.get("epochs", 1.0),
                rank=cfg.get("rank", 64),
                use_qlora=cfg.get("use_qlora", False),
                gpu_name=gpu_name,
                gpu_vram_mb=gpu_vram_mb,
            )
            results.append({
                "config_id": i + 1,
                "label": cfg.get("label", f"Config {i+1}"),
                **cfg,
                "vram_gb": round(est.vram.total_gb, 1),
                "time": est.time_human,
                "steps": est.steps_total,
                "feasible": est.feasible,
                "warnings": est.warnings,
            })
        return results

    def _gpu_throughput(self, gpu_name: str, model_b: float, qlora: bool) -> float:
        """查询 GPU 训练吞吐量 (tokens/sec)"""
        name = gpu_name.lower()
        for pattern in sorted(GPU_THROUGHPUT.keys(), key=len, reverse=True):
            if pattern in name:
                fp16_rate, qlora_rate = GPU_THROUGHPUT[pattern]
                base_rate = qlora_rate if qlora else fp16_rate
                # 吞吐量与模型大小反比
                return base_rate / max(model_b, 0.5)
        return 0

    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f} 秒"
        elif seconds < 3600:
            return f"{seconds/60:.0f} 分钟"
        elif seconds < 86400:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            return f"{h} 小时 {m} 分钟"
        else:
            d = int(seconds // 86400)
            h = int((seconds % 86400) // 3600)
            return f"{d} 天 {h} 小时"


# 便捷函数
_estimator = TrainingEstimator()
estimate_training = _estimator.estimate
recommend_params = _estimator.recommend_params
compare_configs = _estimator.compare_configs


# ================================================================
#  DatasetAnalyzer — 数据集质量分析
# ================================================================

class DatasetAnalyzer:
    """分析训练数据集质量，给出评分和改进建议。"""

    def analyze(self, dataset_path: str) -> Dict:
        """分析数据集，返回质量报告。"""
        import json
        from pathlib import Path

        p = Path(dataset_path)
        if not p.exists():
            return {"error": f"文件不存在: {dataset_path}", "score": 0}

        # 读取数据
        rows = []
        try:
            if p.suffix == ".jsonl":
                for line in p.read_text(encoding="utf-8").strip().splitlines():
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            elif p.suffix == ".json":
                data = json.loads(p.read_text(encoding="utf-8"))
                rows = data if isinstance(data, list) else [data]
            elif p.suffix == ".csv":
                import csv
                with open(p, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
            else:
                return {"error": f"不支持的格式: {p.suffix}（支持 .jsonl, .json, .csv）", "score": 0}
        except Exception as e:
            return {"error": f"读取失败: {e}", "score": 0}

        if not rows:
            return {"error": "数据集为空", "score": 0}

        report = {
            "total_samples": len(rows),
            "file_size_kb": p.stat().st_size / 1024,
            "format": p.suffix,
            "issues": [],
            "suggestions": [],
        }

        # 1. 检测数据格式
        keys = set()
        for r in rows[:100]:
            if isinstance(r, dict):
                keys.update(r.keys())

        report["fields"] = sorted(keys)

        # 判断数据类型
        has_instruction = "instruction" in keys
        has_input = "input" in keys
        has_output = "output" in keys
        has_text = "text" in keys
        has_prompt = "prompt" in keys
        has_chosen = "chosen" in keys
        has_rejected = "rejected" in keys
        has_messages = "messages" in keys or "conversations" in keys

        if has_chosen and has_rejected:
            report["dataset_type"] = "偏好数据 (DPO/ORPO)"
        elif has_instruction and has_output:
            report["dataset_type"] = "指令数据 (SFT)"
        elif has_messages or (has_text and any("###" in str(r.get("text", "")) for r in rows[:10])):
            report["dataset_type"] = "对话数据 (SFT)"
        elif has_text:
            report["dataset_type"] = "纯文本 (续写)"
        else:
            report["dataset_type"] = "未知格式"
            report["issues"].append("⚠️ 无法识别数据格式，可能需要转换")

        # 2. 空值检测
        empty_count = 0
        field_empty = {}
        for r in rows:
            if not isinstance(r, dict):
                continue
            for k, v in r.items():
                if v is None or (isinstance(v, str) and not v.strip()):
                    empty_count += 1
                    field_empty[k] = field_empty.get(k, 0) + 1

        if empty_count > 0:
            empty_pct = empty_count / (len(rows) * max(len(keys), 1)) * 100
            report["empty_fields"] = field_empty
            report["empty_pct"] = round(empty_pct, 1)
            if empty_pct > 10:
                report["issues"].append(f"🔴 空值率 {empty_pct:.1f}%，建议清洗")
            elif empty_pct > 2:
                report["issues"].append(f"🟡 空值率 {empty_pct:.1f}%，可接受但建议检查")

        # 3. 重复检测
        seen = set()
        dup_count = 0
        for r in rows:
            if isinstance(r, dict):
                # 用主要字段做指纹
                fp = ""
                for k in ["instruction", "prompt", "text", "chosen"]:
                    if k in r and r[k]:
                        fp = str(r[k])[:200]
                        break
                if fp:
                    if fp in seen:
                        dup_count += 1
                    seen.add(fp)

        dup_pct = dup_count / max(len(rows), 1) * 100
        report["duplicates"] = dup_count
        report["duplicate_pct"] = round(dup_pct, 1)
        if dup_pct > 10:
            report["issues"].append(f"🔴 重复率 {dup_pct:.1f}%（{dup_count} 条），严重影响训练")
        elif dup_pct > 3:
            report["issues"].append(f"🟡 重复率 {dup_pct:.1f}%（{dup_count} 条），建议去重")

        # 4. 长度分布
        lengths = []
        for r in rows:
            if isinstance(r, dict):
                text = ""
                for k in ["instruction", "output", "text", "prompt", "chosen"]:
                    if k in r and r[k]:
                        text += str(r[k])
                lengths.append(len(text))

        if lengths:
            report["length_stats"] = {
                "min": min(lengths),
                "max": max(lengths),
                "avg": round(sum(lengths) / len(lengths)),
                "median": sorted(lengths)[len(lengths) // 2],
            }
            # 极短样本
            too_short = sum(1 for l in lengths if l < 10)
            if too_short > len(lengths) * 0.1:
                report["issues"].append(f"🟡 {too_short} 条样本过短（<10字符），可能是噪声")
            # 极长样本
            too_long = sum(1 for l in lengths if l > 8000)
            if too_long > 0:
                report["issues"].append(f"🟡 {too_long} 条样本超长（>8000字符），训练时会被截断")

        # 5. 评分
        score = 100
        for issue in report["issues"]:
            if "🔴" in issue:
                score -= 25
            elif "🟡" in issue:
                score -= 10
            elif "⚠️" in issue:
                score -= 15
        score = max(0, score)
        report["score"] = score

        if score >= 90:
            report["grade"] = "A（优秀）"
        elif score >= 75:
            report["grade"] = "B（良好）"
        elif score >= 60:
            report["grade"] = "C（及格）"
        else:
            report["grade"] = "D（需改进）"

        # 建议
        if report["total_samples"] < 100:
            report["suggestions"].append("💡 样本量较少（<100），建议扩充数据或使用数据增强")
        if report["total_samples"] > 50000:
            report["suggestions"].append("💡 样本量较大，可适当减少 epoch 数")
        if not report["issues"]:
            report["suggestions"].append("✨ 数据集质量良好，可直接用于训练")

        return report

    def format_report(self, report: Dict) -> str:
        """格式化报告为可读文本。"""
        if "error" in report:
            return f"❌ {report['error']}"

        lines = []
        lines.append(f"📊 数据集质量报告")
        lines.append(f"{'='*50}")
        lines.append(f"评分: {report.get('score', '?')}/100  等级: {report.get('grade', '?')}")
        lines.append(f"样本数: {report.get('total_samples', '?')}  大小: {report.get('file_size_kb', 0):.1f} KB")
        lines.append(f"类型: {report.get('dataset_type', '?')}")
        lines.append(f"字段: {', '.join(report.get('fields', []))}")

        stats = report.get("length_stats")
        if stats:
            lines.append(f"\n📏 长度分布:")
            lines.append(f"   最短: {stats['min']} | 中位: {stats['median']} | 平均: {stats['avg']} | 最长: {stats['max']}")

        if report.get("duplicates", 0) > 0:
            lines.append(f"\n🔍 重复: {report['duplicates']} 条 ({report.get('duplicate_pct', 0):.1f}%)")

        if report.get("empty_pct", 0) > 0:
            lines.append(f"🕳️ 空值率: {report['empty_pct']:.1f}%")

        issues = report.get("issues", [])
        if issues:
            lines.append(f"\n⚠️ 发现问题:")
            for issue in issues:
                lines.append(f"   {issue}")

        suggestions = report.get("suggestions", [])
        if suggestions:
            lines.append(f"\n💡 建议:")
            for s in suggestions:
                lines.append(f"   {s}")

        return "\n".join(lines)
