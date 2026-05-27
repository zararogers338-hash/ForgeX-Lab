# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

"""ForgeX 锻造流水线 — 自动化迭代训练。

核心流程:
  Data → SFT Round 1 → Benchmark → 分析 → 调参 → SFT Round 2 → ... → 达标 → 合并导出

功能:
  1. 自动迭代: 训练 → 评测 → 判断 → 调参 → 重训
  2. 智能调参: 根据 loss 曲线和评测分数自动调整 lr/rank/epochs
  3. 早停策略: 评测分数达标或连续无提升时停止
  4. 训练历史: 记录每轮参数和结果，支持回溯对比
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from core import LORAS_DIR, DATASETS_DIR, log


@dataclass
class ForgeRound:
    """单轮锻造记录"""
    round_num: int = 0
    params: Dict = field(default_factory=dict)
    lora_path: str = ""
    final_loss: float = float("inf")
    eval_loss: float = float("inf")
    bench_score: float = 0.0
    duration_sec: float = 0.0
    status: str = "pending"  # pending / running / done / failed
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            "round": self.round_num,
            "params": {k: v for k, v in self.params.items() if not k.startswith("_")},
            "lora_path": self.lora_path,
            "final_loss": round(self.final_loss, 4) if self.final_loss < 100 else None,
            "eval_loss": round(self.eval_loss, 4) if self.eval_loss < 100 else None,
            "bench_score": round(self.bench_score, 1),
            "duration": f"{self.duration_sec:.0f}s",
            "status": self.status,
            "notes": self.notes,
        }


@dataclass
class ForgeConfig:
    """锻造配置"""
    max_rounds: int = 3
    target_loss: float = 0.5
    target_bench_score: float = 70.0
    patience: int = 2                      # 连续无提升轮数上限
    auto_adjust_lr: bool = True            # 自动调学习率
    auto_adjust_rank: bool = False         # 自动调 LoRA rank
    run_benchmark: bool = True             # 每轮评测
    merge_on_complete: bool = False        # 达标后自动合并
    # 调参策略
    lr_decay: float = 0.5                  # lr 每轮衰减比
    lr_min: float = 1e-5
    rank_schedule: List[int] = field(default_factory=lambda: [64, 128])  # 逐轮增加 rank


class ForgePipeline:
    """锻造流水线引擎"""

    def __init__(self):
        self.rounds: List[ForgeRound] = []
        self.config = ForgeConfig()
        self._running = False
        self._should_stop = False

    def run(
        self,
        base_model: str,
        dataset_path: Any,
        initial_params: Dict,
        config: ForgeConfig = None,
        task=None,
        train_fn: Callable = None,
        bench_fn: Callable = None,
    ) -> List[ForgeRound]:
        """执行锻造流水线

        Args:
            base_model: 基座模型 ID
            dataset_path: 数据集路径
            initial_params: 初始训练参数
            config: 锻造配置
            task: 任务对象（用于进度更新）
            train_fn: 训练函数 (base_model, dataset, params) -> lora_path
            bench_fn: 评测函数 (model_path) -> score
        """
        if config:
            self.config = config
        self.rounds = []
        self._running = True
        self._should_stop = False

        cfg = self.config
        params = dict(initial_params)
        best_loss = float("inf")
        best_score = 0.0
        no_improve_count = 0

        for round_num in range(1, cfg.max_rounds + 1):
            if self._should_stop:
                log("锻造被手动停止")
                break

            rd = ForgeRound(round_num=round_num, params=dict(params))
            rd.status = "running"
            self.rounds.append(rd)

            _update(task, _pct(round_num, cfg.max_rounds, 0),
                    f"🔥 锻造 Round {round_num}/{cfg.max_rounds}")
            log(f"═══ 锻造 Round {round_num}/{cfg.max_rounds} ═══")
            log(f"  参数: lr={params.get('lr')}, rank={params.get('rank')}, epochs={params.get('epochs')}")

            # ── 训练 ──
            t0 = time.time()
            try:
                rd.params["output_name"] = f"{params.get('output_name', 'forge')}_{round_num}"
                if train_fn:
                    lora_path = train_fn(base_model, dataset_path, rd.params, task=task)
                else:
                    from core.trainer import TrainerEngine
                    engine = TrainerEngine()
                    lora_path = engine._train_sft(base_model, dataset_path, rd.params, task=task)
                rd.lora_path = str(lora_path)
                rd.duration_sec = time.time() - t0

                # 读取 meta 获取 final loss
                meta = _read_meta(lora_path)
                rd.final_loss = meta.get("final_loss", rd.final_loss)

                _update(task, _pct(round_num, cfg.max_rounds, 70),
                        f"Round {round_num} 训练完成 (loss={rd.final_loss:.4f})")
            except Exception as e:
                rd.status = "failed"
                rd.notes = str(e)
                log(f"Round {round_num} 训练失败: {e}")
                break

            # ── 评测 ──
            if cfg.run_benchmark and bench_fn:
                try:
                    _update(task, _pct(round_num, cfg.max_rounds, 80),
                            f"Round {round_num} 评测中...")
                    rd.bench_score = bench_fn(rd.lora_path)
                    log(f"  评测分数: {rd.bench_score:.1f}")
                except Exception as e:
                    log(f"  评测失败（不影响流程）: {e}")

            rd.status = "done"

            # ── 判断是否达标 ──
            improved = False
            if rd.final_loss < best_loss - 0.01:
                best_loss = rd.final_loss
                improved = True
            if rd.bench_score > best_score + 1:
                best_score = rd.bench_score
                improved = True

            if rd.final_loss <= cfg.target_loss:
                rd.notes = f"✅ 达标! loss={rd.final_loss:.4f} <= {cfg.target_loss}"
                log(rd.notes)
                break
            if rd.bench_score >= cfg.target_bench_score and cfg.run_benchmark:
                rd.notes = f"✅ 达标! bench={rd.bench_score:.1f} >= {cfg.target_bench_score}"
                log(rd.notes)
                break

            if not improved:
                no_improve_count += 1
                if no_improve_count >= cfg.patience:
                    rd.notes = f"⏹️ 连续 {cfg.patience} 轮无提升，停止"
                    log(rd.notes)
                    break
            else:
                no_improve_count = 0

            # ── 调参 ──
            if round_num < cfg.max_rounds:
                params = self._adjust_params(params, rd, round_num, cfg)
                log(f"  下一轮参数: lr={params.get('lr')}, rank={params.get('rank')}")

        self._running = False

        # ── 总结 ──
        _update(task, 95, "锻造完成，生成总结...")
        summary = self.summary()
        log(f"═══ 锻造完成 ═══\n{summary}")

        # 保存历史
        self._save_history(base_model)

        _update(task, 100, f"✅ 锻造完成: {len(self.rounds)} 轮")
        return self.rounds

    def stop(self):
        self._should_stop = True

    def _adjust_params(self, params: Dict, rd: ForgeRound, round_num: int, cfg: ForgeConfig) -> Dict:
        """根据当前结果调整下一轮参数"""
        new_p = dict(params)

        # 学习率衰减
        if cfg.auto_adjust_lr:
            current_lr = float(new_p.get("lr", 2e-4))
            if rd.final_loss > 1.0:
                # loss 还很高，维持或略降
                new_lr = current_lr * 0.8
            else:
                # loss 较低，大幅降低 lr 精调
                new_lr = current_lr * cfg.lr_decay
            new_p["lr"] = max(cfg.lr_min, new_lr)

        # Rank 递增
        if cfg.auto_adjust_rank and round_num <= len(cfg.rank_schedule):
            new_p["rank"] = cfg.rank_schedule[min(round_num - 1, len(cfg.rank_schedule) - 1)]

        # 后续轮次增加 epochs
        base_epochs = float(new_p.get("epochs", 1.0))
        if rd.final_loss > 0.8:
            new_p["epochs"] = min(base_epochs * 1.5, 5.0)

        return new_p

    def summary(self) -> str:
        if not self.rounds:
            return "没有训练记录"
        lines = []
        lines.append(f"总轮数: {len(self.rounds)}")
        done_rounds = [r for r in self.rounds if r.status == "done"]
        if done_rounds:
            best = min(done_rounds, key=lambda r: r.final_loss)
            lines.append(f"最佳 loss: {best.final_loss:.4f} (Round {best.round_num})")
            if any(r.bench_score > 0 for r in done_rounds):
                best_bench = max(done_rounds, key=lambda r: r.bench_score)
                lines.append(f"最佳评测: {best_bench.bench_score:.1f} (Round {best_bench.round_num})")
            total_time = sum(r.duration_sec for r in done_rounds)
            lines.append(f"总耗时: {total_time/60:.1f} 分钟")
            lines.append(f"推荐使用: {best.lora_path}")
        return "\n".join(lines)

    def get_history_table(self) -> List[List[str]]:
        """生成历史表格 (for Gradio Dataframe)"""
        rows = []
        for r in self.rounds:
            rows.append([
                str(r.round_num),
                f"{r.params.get('lr', '?')}",
                str(r.params.get("rank", "?")),
                f"{r.final_loss:.4f}" if r.final_loss < 100 else "-",
                f"{r.bench_score:.1f}" if r.bench_score > 0 else "-",
                f"{r.duration_sec:.0f}s",
                r.status,
                r.notes[:40],
            ])
        return rows

    def _save_history(self, base_model: str):
        """保存锻造历史"""
        history_dir = Path(LORAS_DIR) / ".forge_history"
        history_dir.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = history_dir / f"forge_{ts}.json"
        data = {
            "base_model": base_model,
            "timestamp": ts,
            "config": {
                "max_rounds": self.config.max_rounds,
                "target_loss": self.config.target_loss,
                "target_bench_score": self.config.target_bench_score,
            },
            "rounds": [r.to_dict() for r in self.rounds],
            "summary": self.summary(),
        }
        try:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            log(f"保存锻造历史失败: {e}")


# ── 训练历史浏览器 ──

def list_training_history() -> List[Dict]:
    """扫描所有 LoRA 目录，读取 forgex_meta.json"""
    results = []
    for d in Path(LORAS_DIR).iterdir():
        if not d.is_dir():
            continue
        meta_path = d / "forgex_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["lora_name"] = d.name
                meta["lora_path"] = str(d)
                results.append(meta)
            except Exception:
                pass
    results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return results


def list_forge_history() -> List[Dict]:
    """列出锻造流水线历史"""
    history_dir = Path(LORAS_DIR) / ".forge_history"
    if not history_dir.exists():
        return []
    results = []
    for f in sorted(history_dir.glob("forge_*.json"), reverse=True):
        try:
            results.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return results


def format_history_table(history: List[Dict]) -> List[List[str]]:
    """格式化训练历史为表格"""
    rows = []
    for h in history:
        rows.append([
            h.get("lora_name", "-"),
            h.get("base_model", "-")[:30],
            h.get("method", "sft"),
            str(h.get("dataset_size", "-")),
            str(h.get("rank", "-")),
            str(h.get("lr", "-")),
            "✅" if h.get("unsloth") else "❌",
            h.get("timestamp", "-"),
        ])
    return rows


# ── 工具函数 ──

def _update(task, pct, msg):
    if task:
        try:
            task.update_progress(float(pct), str(msg))
        except Exception:
            pass

def _pct(round_num, max_rounds, phase_pct):
    """计算总进度百分比"""
    per_round = 90.0 / max(max_rounds, 1)
    return 5 + (round_num - 1) * per_round + phase_pct * per_round / 100

def _read_meta(lora_path) -> Dict:
    p = Path(lora_path) / "forgex_meta.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


# 单例
forge_pipeline = ForgePipeline()
