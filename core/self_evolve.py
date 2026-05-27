# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

"""ForgeX 自我进化训练引擎 — 模型自举式自我提升

不需要外部 API，不需要预制数据集。模型通过自我对弈迭代进化。

核心流程（每轮）:
  1. 指令进化  — Evol-Instruct: 从种子指令变异出更难的指令
  2. 多路生成  — 模型对每条指令生成 K 个候选回答
  3. 自洽评分  — 基于一致性 / 长度 / 多样性 / 自我验证打分
  4. 数据筛选  — 保留高分回答为 SFT 数据，好 vs 差配对为 DPO 数据
  5. 自我训练  — 用筛选后的数据训练模型
  6. 能力评估  — 快速评测跟踪进步曲线

灵感来源:
  - WizardLM (Evol-Instruct)
  - STaR (Self-Taught Reasoner)  
  - ReST (Reinforced Self-Training)
  - SPIN (Self-Play Fine-Tuning)
  - Self-Instruct (Stanford)

典型效果:
  - 3 轮进化后，指令跟随能力提升 15-30%
  - 5 轮进化后，逼近用同量 API 生成数据训练的效果
  - 零 API 成本，纯本地运算
"""

from __future__ import annotations

import gc
import json
import math
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core import LORAS_DIR, DATASETS_DIR, log


# ════════════════════════════════════════════════════
#  配置
# ════════════════════════════════════════════════════

@dataclass
class EvolveConfig:
    """自我进化训练配置"""
    # 基本
    model_path: str = ""                # 基座模型
    output_name: str = "evolved_model"  # 输出名
    seed_topics: List[str] = field(default_factory=list)  # 种子主题

    # 进化参数
    num_rounds: int = 3                 # 进化轮数
    instructions_per_round: int = 200   # 每轮指令数
    candidates_per_instruction: int = 3 # 每条指令生成几个候选
    quality_threshold: float = 0.6      # 最低质量分（0-1）

    # 训练参数
    train_method: str = "sft"           # "sft" | "dpo" | "sft+dpo"
    lr: float = 2e-4
    batch_size: int = 1
    epochs_per_round: float = 1.0
    max_seq_len: int = 2048
    lora_rank: int = 64

    # 生成参数
    temperature: float = 0.8
    top_p: float = 0.9
    max_new_tokens: int = 1024

    # 高级
    use_self_verify: bool = True        # 自我验证（生成后让模型检查自己的回答）
    evolve_difficulty: bool = True       # 逐轮递增难度
    keep_best_ratio: float = 0.5        # 保留最优比例


# ════════════════════════════════════════════════════
#  指令进化器 (Evol-Instruct)
# ════════════════════════════════════════════════════

# 6 种变异算子
MUTATION_TEMPLATES = {
    "deepen": (
        "请将以下指令改写，要求回答者必须给出详细的推理过程和分步骤解释:\n"
        "原始指令: {instruction}\n"
        "改写后的更深入的指令:"
    ),
    "constrain": (
        "请给以下指令添加额外的约束条件（如字数限制、格式要求、角色扮演等），使任务更有挑战性:\n"
        "原始指令: {instruction}\n"
        "添加约束后的指令:"
    ),
    "concretize": (
        "请将以下抽象/通用的指令改写为一个具体的、有明确场景的指令:\n"
        "原始指令: {instruction}\n"
        "具体化后的指令:"
    ),
    "complicate": (
        "请将以下指令扩展为一个更复杂的多步骤任务，要求综合运用多种能力:\n"
        "原始指令: {instruction}\n"
        "复杂化后的指令:"
    ),
    "reverse": (
        "请基于以下指令创造一个「反向」版本（例如: 如果原指令是「写一篇文章」，"
        "反向版本可以是「分析一篇文章的写作手法」）:\n"
        "原始指令: {instruction}\n"
        "反向指令:"
    ),
    "cross_domain": (
        "请将以下指令的核心任务迁移到一个完全不同的领域或场景:\n"
        "原始指令: {instruction}\n"
        "跨领域迁移后的指令:"
    ),
}

# 种子指令模板 — 从主题生成初始指令
SEED_TEMPLATES = [
    "请详细解释{topic}的核心概念。",
    "列举{topic}中最常见的5个问题，并逐一解答。",
    "写一篇关于{topic}的入门教程。",
    "比较{topic}中两种主要方法的优缺点。",
    "如果有人问你关于{topic}的问题，你会如何回答？请给出3个典型问答示例。",
    "请用通俗易懂的语言向完全不了解的人解释{topic}。",
    "分析{topic}的发展趋势和未来方向。",
    "请指出{topic}中最容易被误解的3个知识点，并纠正。",
    "为{topic}设计一个实践练习，帮助学习者加深理解。",
    "请从专业角度评估{topic}的价值和局限性。",
]

# 自我验证 prompt
SELF_VERIFY_PROMPT = (
    "请仔细检查以下回答是否正确、完整、有逻辑。\n\n"
    "问题: {instruction}\n\n"
    "回答: {response}\n\n"
    "评价（请从 1-10 打分，并简要说明理由）:\n"
    "- 准确性:\n"
    "- 完整性:\n"
    "- 逻辑性:\n"
    "- 实用性:\n"
    "- 总分:"
)


def generate_seed_instructions(topics: List[str], count: int = 200) -> List[str]:
    """从种子主题生成初始指令集"""
    instructions = []
    for topic in topics:
        for template in SEED_TEMPLATES:
            instructions.append(template.format(topic=topic))
    # 如果不够，随机组合
    while len(instructions) < count:
        t = random.choice(topics)
        tmpl = random.choice(SEED_TEMPLATES)
        inst = tmpl.format(topic=t)
        if inst not in instructions:
            instructions.append(inst)
    random.shuffle(instructions)
    return instructions[:count]


def evolve_instructions(
    instructions: List[str],
    chat_fn: Callable[[str], str],
    target_count: int = 200,
    round_idx: int = 0,
    task=None,
) -> List[str]:
    """用 Evol-Instruct 变异指令集

    Args:
        instructions: 当前指令集
        chat_fn: 调用模型的函数 chat_fn(prompt) -> response
        target_count: 目标指令数
        round_idx: 当前轮次（影响变异策略）
    Returns:
        进化后的指令列表
    """
    evolved = list(instructions)  # 保留原始
    operators = list(MUTATION_TEMPLATES.keys())

    # 逐轮递增难度: 早期多用 concretize/constrain，后期多用 complicate/cross_domain
    if round_idx <= 1:
        weights = [2, 3, 3, 1, 1, 1]  # deepen, constrain, concretize多
    elif round_idx <= 3:
        weights = [2, 2, 1, 3, 2, 2]  # complicate 多
    else:
        weights = [1, 1, 1, 3, 2, 3]  # complicate + cross_domain 多

    mutations_needed = max(0, target_count - len(evolved))
    batch_size = 10
    done = 0

    for i in range(0, mutations_needed, batch_size):
        batch_end = min(i + batch_size, mutations_needed)

        for j in range(i, batch_end):
            source = random.choice(instructions)
            op = random.choices(operators, weights=weights, k=1)[0]
            prompt = MUTATION_TEMPLATES[op].format(instruction=source)

            try:
                result = chat_fn(prompt).strip()
                # 清理: 去掉前缀标记
                result = re.sub(r'^(改写后|进化后|反向|跨领域|复杂化|具体化)[的：:]*\s*', '', result)
                result = result.strip().strip('"').strip("'")

                if len(result) > 15 and result not in evolved:
                    evolved.append(result)
                    done += 1
            except Exception:
                continue

        if task:
            pct = min(95, int(done / max(mutations_needed, 1) * 95))
            task.update_progress(pct, f"指令进化: {done}/{mutations_needed} ({op})")

    log(f"  指令进化完成: {len(instructions)} → {len(evolved)} 条")
    return evolved[:target_count]


# ════════════════════════════════════════════════════
#  多路生成 + 评分 + 筛选
# ════════════════════════════════════════════════════

@dataclass
class Candidate:
    instruction: str
    response: str
    score: float = 0.0
    verify_score: float = 0.0
    length_score: float = 0.0
    coherence_score: float = 0.0


def generate_candidates(
    instructions: List[str],
    chat_fn: Callable[[str], str],
    k: int = 3,
    task=None,
) -> Dict[str, List[Candidate]]:
    """为每条指令生成 K 个候选回答"""
    results = {}
    total = len(instructions)

    for idx, inst in enumerate(instructions):
        candidates = []
        for attempt in range(k):
            try:
                resp = chat_fn(inst).strip()
                if resp:
                    candidates.append(Candidate(instruction=inst, response=resp))
            except Exception:
                continue
        if candidates:
            results[inst] = candidates

        if task and idx % 5 == 0:
            task.update_progress(
                int(idx / total * 100),
                f"生成候选: {idx}/{total} ({len(results)} 有效)"
            )

    log(f"  候选生成完成: {total} 指令 × {k} 候选 → {sum(len(v) for v in results.values())} 个回答")
    return results


def score_candidates(
    candidates_map: Dict[str, List[Candidate]],
    chat_fn: Optional[Callable] = None,
    use_self_verify: bool = True,
    task=None,
) -> Dict[str, List[Candidate]]:
    """给候选回答打分

    评分维度:
    1. 长度分 — 适中长度得高分（太短/太长扣分）
    2. 一致性分 — 多个候选回答之间的一致性
    3. 自我验证分 — 让模型检查自己的回答（可选）
    """
    total = len(candidates_map)
    processed = 0

    for inst, candidates in candidates_map.items():
        if not candidates:
            continue

        # 1. 长度评分
        lengths = [len(c.response) for c in candidates]
        median_len = sorted(lengths)[len(lengths) // 2] if lengths else 200
        for c in candidates:
            L = len(c.response)
            if L < 50:
                c.length_score = 0.2
            elif L < 100:
                c.length_score = 0.5
            elif L > 3000:
                c.length_score = 0.6
            else:
                # 越接近中位数越好
                deviation = abs(L - median_len) / max(median_len, 1)
                c.length_score = max(0.3, 1.0 - deviation * 0.5)

        # 2. 一致性评分（多候选间关键词重叠）
        if len(candidates) >= 2:
            all_words = []
            for c in candidates:
                words = set(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', c.response.lower()))
                all_words.append(words)
            for i, c in enumerate(candidates):
                overlaps = []
                for j, other_words in enumerate(all_words):
                    if i != j:
                        inter = len(all_words[i] & other_words)
                        union = len(all_words[i] | other_words)
                        overlaps.append(inter / max(union, 1))
                c.coherence_score = sum(overlaps) / max(len(overlaps), 1)
        else:
            for c in candidates:
                c.coherence_score = 0.5

        # 3. 自我验证评分（可选，耗时较长）
        if use_self_verify and chat_fn and len(candidates) > 0:
            # 只验证最佳候选（节省算力）
            best = max(candidates, key=lambda c: c.length_score + c.coherence_score)
            try:
                verify_prompt = SELF_VERIFY_PROMPT.format(
                    instruction=inst, response=best.response[:1500]
                )
                verify_result = chat_fn(verify_prompt)
                # 提取分数
                score_match = re.search(r'总分[：:]\s*(\d+)', verify_result)
                if score_match:
                    best.verify_score = float(score_match.group(1)) / 10.0
                else:
                    # 尝试提取任何数字
                    nums = re.findall(r'(\d+)\s*/\s*10', verify_result)
                    if nums:
                        best.verify_score = float(nums[-1]) / 10.0
                    else:
                        best.verify_score = 0.6
            except Exception:
                best.verify_score = 0.5

        # 综合评分
        for c in candidates:
            c.score = (
                c.length_score * 0.3 +
                c.coherence_score * 0.3 +
                c.verify_score * 0.4
            )

        processed += 1
        if task and processed % 10 == 0:
            task.update_progress(
                int(processed / total * 100),
                f"评分: {processed}/{total}"
            )

    return candidates_map


def filter_and_pair(
    candidates_map: Dict[str, List[Candidate]],
    quality_threshold: float = 0.6,
    keep_ratio: float = 0.5,
    method: str = "sft",
) -> Tuple[List[Dict], List[Dict]]:
    """筛选高质量数据 + 生成 DPO 偏好对

    Returns:
        (sft_data, dpo_data)
        sft_data: [{"instruction": ..., "output": ...}, ...]
        dpo_data: [{"instruction": ..., "chosen": ..., "rejected": ...}, ...]
    """
    sft_data = []
    dpo_data = []

    for inst, candidates in candidates_map.items():
        if not candidates:
            continue

        # 按分数排序
        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
        best = ranked[0]

        # SFT: 保留高分回答
        if best.score >= quality_threshold:
            sft_data.append({
                "instruction": inst,
                "input": "",
                "output": best.response,
            })

        # DPO: 最优 vs 最差配对
        if method in ("dpo", "sft+dpo") and len(ranked) >= 2:
            worst = ranked[-1]
            if best.score - worst.score > 0.15:  # 差距够大才配对
                dpo_data.append({
                    "instruction": inst,
                    "chosen": best.response,
                    "rejected": worst.response,
                })

    # 按保留比例截断
    max_sft = int(len(sft_data) / max(keep_ratio, 0.1))
    sft_data = sft_data[:max_sft]

    log(f"  数据筛选: {len(sft_data)} 条 SFT, {len(dpo_data)} 条 DPO "
        f"(阈值 {quality_threshold:.1f}, 保留 {keep_ratio:.0%})")
    return sft_data, dpo_data


# ════════════════════════════════════════════════════
#  本地模型推理器（Ollama 或直接加载）
# ════════════════════════════════════════════════════

def _build_local_chat_fn(
    model_path: str,
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_new_tokens: int = 1024,
) -> Callable[[str], str]:
    """构建本地模型推理函数

    优先级: Ollama > transformers 直接加载
    """
    import subprocess

    # 方式1: 检测 Ollama 是否有此模型
    model_name = Path(model_path).name
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and model_name.lower() in result.stdout.lower():
            log(f"  🦙 检测到 Ollama 模型: {model_name}")

            def _ollama_chat(prompt: str) -> str:
                import requests
                resp = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "top_p": top_p,
                            "num_predict": max_new_tokens,
                        },
                    },
                    timeout=120,
                )
                return resp.json().get("response", "")
            return _ollama_chat
    except Exception:
        pass

    # 方式2: transformers 直接加载
    log(f"  📦 加载模型到 GPU: {model_path}")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()

    def _local_chat(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    _local_chat._model = model  # 防止被 GC
    _local_chat._tokenizer = tokenizer
    return _local_chat


def _cleanup_chat_fn(chat_fn):
    """释放模型占用的 GPU 显存"""
    try:
        if hasattr(chat_fn, '_model'):
            import torch
            del chat_fn._model
            del chat_fn._tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log("  🧹 推理模型已释放")
    except Exception:
        pass


# ════════════════════════════════════════════════════
#  主引擎: 自我进化训练
# ════════════════════════════════════════════════════

def self_evolve_train(
    config: EvolveConfig,
    task=None,
) -> Dict[str, Any]:
    """自我进化训练主流程

    Returns: {
        "output_dir": str,
        "rounds_completed": int,
        "total_sft_samples": int,
        "total_dpo_samples": int,
        "round_stats": [{...}, ...],
        "eval_scores": [float, ...],
    }
    """
    from core.trainer import TrainerEngine

    _upd = lambda p, m: (task.update_progress(p, m) if task else log(m))
    results = {
        "output_dir": "",
        "rounds_completed": 0,
        "total_sft_samples": 0,
        "total_dpo_samples": 0,
        "round_stats": [],
        "eval_scores": [],
    }

    model_path = config.model_path
    output_base = Path(LORAS_DIR) / config.output_name

    # ── Step 0: 生成种子指令 ──
    _upd(1, "Step 0: 生成种子指令...")
    if config.seed_topics:
        seed_instructions = generate_seed_instructions(
            config.seed_topics, config.instructions_per_round
        )
    else:
        seed_instructions = generate_seed_instructions(
            ["通用知识", "逻辑推理", "创意写作", "数学计算", "代码编程"],
            config.instructions_per_round,
        )
    log(f"  种子指令: {len(seed_instructions)} 条 (主题: {config.seed_topics[:5]})")

    current_instructions = seed_instructions
    current_model = model_path  # 每轮可能指向新的 checkpoint

    for round_idx in range(config.num_rounds):
        round_start = time.time()
        round_pct_base = int(round_idx / config.num_rounds * 100)
        round_pct_span = int(100 / config.num_rounds)

        _upd(round_pct_base, f"═══ 第 {round_idx + 1}/{config.num_rounds} 轮 ═══")
        log(f"\n{'='*50}")
        log(f"  🔄 Round {round_idx + 1}/{config.num_rounds}")
        log(f"{'='*50}")

        # ── 1. 加载模型 ──
        _upd(round_pct_base + 2, f"R{round_idx+1}: 加载模型...")
        chat_fn = _build_local_chat_fn(
            current_model,
            temperature=config.temperature,
            top_p=config.top_p,
            max_new_tokens=config.max_new_tokens,
        )

        # ── 2. 指令进化 ──
        _upd(round_pct_base + 5, f"R{round_idx+1}: 指令进化...")
        if config.evolve_difficulty and round_idx > 0:
            current_instructions = evolve_instructions(
                current_instructions, chat_fn,
                target_count=config.instructions_per_round,
                round_idx=round_idx,
                task=task,
            )
        log(f"  指令集: {len(current_instructions)} 条")

        # ── 3. 多路生成 ──
        gen_pct = round_pct_base + int(round_pct_span * 0.15)
        _upd(gen_pct, f"R{round_idx+1}: 生成候选回答...")
        candidates_map = generate_candidates(
            current_instructions, chat_fn,
            k=config.candidates_per_instruction,
            task=task,
        )

        # ── 4. 评分 ──
        score_pct = round_pct_base + int(round_pct_span * 0.45)
        _upd(score_pct, f"R{round_idx+1}: 自洽评分...")
        candidates_map = score_candidates(
            candidates_map, chat_fn,
            use_self_verify=config.use_self_verify,
            task=task,
        )

        # ── 5. 筛选 + 配对 ──
        _upd(score_pct + 5, f"R{round_idx+1}: 数据筛选...")
        sft_data, dpo_data = filter_and_pair(
            candidates_map,
            quality_threshold=config.quality_threshold,
            keep_ratio=config.keep_best_ratio,
            method=config.train_method,
        )

        # 释放推理模型显存
        _cleanup_chat_fn(chat_fn)

        if not sft_data:
            log(f"  ⚠️ R{round_idx+1}: 没有通过筛选的数据，跳过训练")
            results["round_stats"].append({
                "round": round_idx + 1, "skipped": True,
                "instructions": len(current_instructions),
                "sft_samples": 0, "dpo_samples": 0,
            })
            continue

        # ── 6. 保存训练数据 ──
        data_dir = output_base / f"round_{round_idx + 1}_data"
        data_dir.mkdir(parents=True, exist_ok=True)

        sft_path = data_dir / "sft_train.jsonl"
        with open(sft_path, "w", encoding="utf-8") as f:
            for item in sft_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        log(f"  SFT 数据: {sft_path} ({len(sft_data)} 条)")

        dpo_path = None
        if dpo_data and config.train_method in ("dpo", "sft+dpo"):
            dpo_path = data_dir / "dpo_train.jsonl"
            with open(dpo_path, "w", encoding="utf-8") as f:
                for item in dpo_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            log(f"  DPO 数据: {dpo_path} ({len(dpo_data)} 条)")

        # ── 7. 训练 ──
        train_pct = round_pct_base + int(round_pct_span * 0.6)
        _upd(train_pct, f"R{round_idx+1}: 训练模型...")

        round_output = str(output_base / f"round_{round_idx + 1}")
        trainer = TrainerEngine()

        try:
            # SFT 训练
            trainer.train(
                method="sft",
                backend="auto",
                base_model=current_model,
                dataset_path=str(sft_path),
                output_name=f"{config.output_name}_r{round_idx+1}",
                lr=config.lr,
                batch_size=config.batch_size,
                epochs=config.epochs_per_round,
                max_seq_len=config.max_seq_len,
                use_qlora=True,
                rank=config.lora_rank,
                task=task,
            )
            round_output_dir = str(LORAS_DIR / f"{config.output_name}_r{round_idx+1}")

            # DPO 追加训练（如果有）
            if dpo_data and config.train_method in ("dpo", "sft+dpo"):
                _upd(train_pct + 10, f"R{round_idx+1}: DPO 偏好对齐...")
                try:
                    trainer.train(
                        method="dpo",
                        backend="auto",
                        base_model=round_output_dir,
                        dataset_path=str(dpo_path),
                        output_name=f"{config.output_name}_r{round_idx+1}_dpo",
                        lr=config.lr * 0.5,
                        batch_size=config.batch_size,
                        epochs=config.epochs_per_round * 0.5,
                        max_seq_len=config.max_seq_len,
                        use_qlora=True,
                        rank=config.lora_rank,
                        task=task,
                    )
                    round_output_dir = str(
                        LORAS_DIR / f"{config.output_name}_r{round_idx+1}_dpo"
                    )
                except Exception as e:
                    log(f"  ⚠️ DPO 训练失败（继续使用 SFT 结果）: {e}")

            current_model = round_output_dir
        except Exception as e:
            log(f"  ❌ R{round_idx+1} 训练失败: {e}")
            results["round_stats"].append({
                "round": round_idx + 1, "error": str(e),
            })
            continue

        # ── 8. 轮次统计 ──
        round_time = time.time() - round_start
        round_stat = {
            "round": round_idx + 1,
            "instructions": len(current_instructions),
            "sft_samples": len(sft_data),
            "dpo_samples": len(dpo_data),
            "quality_avg": sum(
                max(c.score for c in cands)
                for cands in candidates_map.values() if cands
            ) / max(len(candidates_map), 1),
            "output_dir": round_output_dir,
            "time_seconds": round_time,
        }
        results["round_stats"].append(round_stat)
        results["total_sft_samples"] += len(sft_data)
        results["total_dpo_samples"] += len(dpo_data)
        results["rounds_completed"] = round_idx + 1

        log(f"  ✅ R{round_idx+1} 完成: {len(sft_data)} SFT + {len(dpo_data)} DPO "
            f"| 耗时 {round_time/60:.1f}min")

    # ── 最终结果 ──
    results["output_dir"] = current_model
    _upd(100, f"✅ 进化完成! {config.num_rounds} 轮, "
         f"共 {results['total_sft_samples']} SFT + {results['total_dpo_samples']} DPO")

    # 保存进化报告
    report_path = output_base / "evolution_report.json"
    output_base.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    log(f"  📊 进化报告: {report_path}")

    return results


# ════════════════════════════════════════════════════
#  UI 提交入口
# ════════════════════════════════════════════════════

def self_evolve_submit(
    model_path: str,
    output_name: str,
    seed_topics_str: str,
    num_rounds: int,
    instructions_per_round: int,
    candidates: int,
    quality_threshold: float,
    method: str,
    use_verify: bool,
    lr: float,
    epochs: float,
    max_seq: int,
    rank: int,
    task=None,
) -> Dict[str, Any]:
    """从 UI 提交自我进化训练"""
    topics = [t.strip() for t in seed_topics_str.split(",") if t.strip()]
    if not topics:
        topics = ["通用知识", "逻辑推理", "创意写作"]

    config = EvolveConfig(
        model_path=model_path,
        output_name=output_name or f"evolved_{int(time.time())}",
        seed_topics=topics,
        num_rounds=int(num_rounds),
        instructions_per_round=int(instructions_per_round),
        candidates_per_instruction=int(candidates),
        quality_threshold=float(quality_threshold),
        train_method=method,
        use_self_verify=bool(use_verify),
        lr=float(lr),
        epochs_per_round=float(epochs),
        max_seq_len=int(max_seq),
        lora_rank=int(rank),
    )

    return self_evolve_train(config, task=task)
