# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

"""ForgeX 数据质量流水线 — 训练前自动体检 + 清洗。

功能:
  1. 精确去重（instruction hash）
  2. 模糊去重（n-gram Jaccard 相似度）
  3. 长度异常过滤（太短/太长）
  4. 空答案/重复答案检测
  5. 质量评分 + 体检报告
  6. 小数据集警告 + 增强建议
"""
import json
import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core import DATASETS_DIR, log


# ════════════════════════════════════════════
#  数据加载
# ════════════════════════════════════════════

def _load_data(path: str) -> List[Dict]:
    """加载 JSON/JSONL 数据文件"""
    p = Path(path)
    if not p.exists():
        p = Path(DATASETS_DIR) / path
    if not p.exists():
        return []

    if p.is_dir():
        items = []
        for f in sorted(p.glob("*.json")) + sorted(p.glob("*.jsonl")):
            items.extend(_load_single_file(f))
        return items
    return _load_single_file(p)


def _load_single_file(p: Path) -> List[Dict]:
    try:
        text = p.read_text(encoding="utf-8-sig").strip()
    except Exception:
        return []
    if not text:
        return []

    # JSON array
    if text.startswith("["):
        try:
            data = json.loads(text)
            return [r for r in data if isinstance(r, dict)]
        except json.JSONDecodeError:
            pass

    # JSONL
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                items.append(obj)
        except json.JSONDecodeError:
            continue
    return items


# ════════════════════════════════════════════
#  去重
# ════════════════════════════════════════════

def _get_text_key(item: Dict) -> str:
    """提取用于去重的主文本"""
    for k in ("instruction", "prompt", "question", "text"):
        v = item.get(k)
        if v and isinstance(v, str) and v.strip():
            return v.strip()
    # messages 格式
    msgs = item.get("messages") or item.get("conversations")
    if isinstance(msgs, list):
        parts = []
        for m in msgs:
            if isinstance(m, dict):
                parts.append(str(m.get("content", "")))
        return " ".join(parts)
    return json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)


def _hash_text(text: str) -> str:
    return hashlib.md5(text.lower().strip().encode("utf-8")).hexdigest()


def _ngrams(text: str, n: int = 3) -> set:
    """提取字符级 n-gram"""
    text = re.sub(r'\s+', ' ', text.lower().strip())
    if len(text) < n:
        return {text}
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def _jaccard(s1: set, s2: set) -> float:
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def dedup_exact(data: List[Dict]) -> Tuple[List[Dict], int]:
    """精确去重（基于 instruction hash）"""
    seen = set()
    clean = []
    dupes = 0
    for item in data:
        h = _hash_text(_get_text_key(item))
        if h in seen:
            dupes += 1
        else:
            seen.add(h)
            clean.append(item)
    return clean, dupes


def dedup_fuzzy(data: List[Dict], threshold: float = 0.85) -> Tuple[List[Dict], int]:
    """模糊去重（n-gram Jaccard > threshold 视为重复）

    为避免 O(n²) 复杂度，只对相邻 50 条做比较（假设数据大致有序）。
    """
    if len(data) <= 1:
        return data, 0

    ngram_cache = [_ngrams(_get_text_key(item)) for item in data]
    keep = [True] * len(data)
    removed = 0
    window = 50  # 只和前 50 条比较

    for i in range(1, len(data)):
        if not keep[i]:
            continue
        for j in range(max(0, i - window), i):
            if not keep[j]:
                continue
            sim = _jaccard(ngram_cache[i], ngram_cache[j])
            if sim >= threshold:
                keep[i] = False
                removed += 1
                break

    clean = [item for item, k in zip(data, keep) if k]
    return clean, removed


# ════════════════════════════════════════════
#  过滤
# ════════════════════════════════════════════

def filter_by_length(data: List[Dict], min_len: int = 10, max_len: int = 50000) -> Tuple[List[Dict], Dict]:
    """按长度过滤"""
    clean = []
    stats = {"too_short": 0, "too_long": 0, "ok": 0}
    for item in data:
        text = _get_text_key(item)
        output = _extract_output(item)
        total = len(text) + len(output)
        if total < min_len:
            stats["too_short"] += 1
        elif total > max_len:
            stats["too_long"] += 1
        else:
            stats["ok"] += 1
            clean.append(item)
    return clean, stats


def filter_empty_output(data: List[Dict]) -> Tuple[List[Dict], int]:
    """过滤空答案"""
    clean = []
    removed = 0
    for item in data:
        output = _extract_output(item)
        if not output or len(output) < 3:
            removed += 1
        else:
            clean.append(item)
    return clean, removed


def detect_repetitive_output(data: List[Dict], threshold: float = 0.15) -> List[Dict]:
    """检测高频重复答案（可能是模型塌缩/数据错误）

    Returns: 重复率超过 threshold 的 (答案, 出现次数, 占比) 列表
    """
    outputs = []
    for item in data:
        out = _extract_output(item)
        if out:
            outputs.append(out[:200])  # 前200字符

    if not outputs:
        return []

    counter = Counter(outputs)
    total = len(outputs)
    repeats = []
    for text, count in counter.most_common(20):
        ratio = count / total
        if ratio >= threshold and count >= 3:
            repeats.append({
                "answer": text[:100] + ("..." if len(text) > 100 else ""),
                "count": count,
                "ratio": round(ratio * 100, 1),
            })
    return repeats


# ════════════════════════════════════════════
#  质量评分
# ════════════════════════════════════════════

def _extract_output(item: Dict) -> str:
    """提取答案/回复文本（支持所有格式）"""
    for k in ("output", "response", "answer", "completion", "chosen"):
        v = item.get(k)
        if v and isinstance(v, str) and v.strip():
            return v.strip()
    # messages / conversations 格式
    msgs = item.get("messages") or item.get("conversations")
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") in ("assistant", "gpt"):
                c = str(m.get("content") or m.get("value") or "").strip()
                if c:
                    return c
    return ""


def _score_item(item: Dict) -> float:
    """对单条数据评分 0-10"""
    score = 5.0  # 基准
    text = _get_text_key(item)
    output = _extract_output(item)

    # 指令质量
    if len(text) < 10:
        score -= 2
    elif len(text) > 50:
        score += 0.5

    # 答案质量 — 注意: 长度判断从大到小（修复: >500 必须在 >100 之前）
    if not output or len(output) < 5:
        score -= 3
    elif len(output) > 500:
        score += 1.5
    elif len(output) > 100:
        score += 1
    elif len(output) > 30:
        score += 0.3

    # 多样性加分（包含代码/数字/特殊格式）
    if any(c in output for c in ("```", "def ", "class ", "import ")):
        score += 0.5
    if re.search(r'\d+', output):
        score += 0.3

    # 结构化格式加分
    if any(c in output for c in ("1.", "- ", "* ", "## ")):
        score += 0.3

    # 重复内容扣分
    if output and text:
        text_words = set(text.split())
        if text_words:
            overlap = len(text_words & set(output.split())) / len(text_words)
            if overlap > 0.8:
                score -= 1.5  # 答案和问题太像

    # 纯符号/乱码扣分
    if output:
        alpha_ratio = sum(c.isalpha() or '\u4e00' <= c <= '\u9fff' for c in output) / max(len(output), 1)
        if alpha_ratio < 0.3:
            score -= 1.5  # 大部分是符号/乱码

    return max(0, min(10, score))


# ════════════════════════════════════════════
#  健康报告
# ════════════════════════════════════════════

def analyze_dataset(path: str) -> Dict[str, Any]:
    """数据集全面体检 — 不修改原数据

    Returns:
        {
            "total": int,
            "format": str,  # "alpaca" / "sharegpt" / "text" / "dpo" / "mixed"
            "exact_dupes": int,
            "fuzzy_dupes": int,
            "empty_outputs": int,
            "length_stats": {"min": int, "max": int, "avg": float, "median": float},
            "too_short": int,
            "too_long": int,
            "repetitive_answers": list,
            "quality_score": float,  # 0-10
            "score_distribution": dict,  # {bucket: count}
            "warnings": list[str],
            "suggestions": list[str],
        }
    """
    data = _load_data(path)
    if not data:
        return {"total": 0, "error": "无法加载数据或数据为空", "warnings": ["数据加载失败"]}

    report: Dict[str, Any] = {"total": len(data)}

    # 格式检测
    keys = set()
    for item in data[:20]:
        keys.update(item.keys())
    if "messages" in keys or "conversations" in keys:
        report["format"] = "sharegpt"
    elif "instruction" in keys and "output" in keys:
        report["format"] = "alpaca"
    elif "prompt" in keys and "chosen" in keys:
        report["format"] = "dpo"
    elif "text" in keys:
        report["format"] = "text"
    else:
        report["format"] = "mixed"

    # 去重分析
    _, exact_dupes = dedup_exact(data)
    report["exact_dupes"] = exact_dupes
    if len(data) <= 5000:  # 大数据集跳过模糊去重（太慢）
        _, fuzzy_dupes = dedup_fuzzy(data)
        report["fuzzy_dupes"] = fuzzy_dupes
    else:
        report["fuzzy_dupes"] = -1  # 跳过标记

    # 空答案
    _, empty = filter_empty_output(data)
    report["empty_outputs"] = empty

    # 长度统计
    lengths = []
    for item in data:
        text = _get_text_key(item)
        output = _extract_output(item)
        lengths.append(len(text) + len(output))
    lengths.sort()
    report["length_stats"] = {
        "min": lengths[0] if lengths else 0,
        "max": lengths[-1] if lengths else 0,
        "avg": round(sum(lengths) / max(len(lengths), 1), 1),
        "median": lengths[len(lengths) // 2] if lengths else 0,
    }

    _, len_stats = filter_by_length(data)
    report["too_short"] = len_stats["too_short"]
    report["too_long"] = len_stats["too_long"]

    # 重复答案
    report["repetitive_answers"] = detect_repetitive_output(data)

    # 质量评分
    scores = [_score_item(item) for item in data]
    report["quality_score"] = round(sum(scores) / max(len(scores), 1), 1)
    buckets = {"优(8-10)": 0, "良(6-8)": 0, "中(4-6)": 0, "差(0-4)": 0}
    for s in scores:
        if s >= 8:
            buckets["优(8-10)"] += 1
        elif s >= 6:
            buckets["良(6-8)"] += 1
        elif s >= 4:
            buckets["中(4-6)"] += 1
        else:
            buckets["差(0-4)"] += 1
    report["score_distribution"] = buckets

    # ---- 生成警告和建议 ----
    warnings = []
    suggestions = []

    if exact_dupes > 0:
        warnings.append(f"⚠️ 精确重复 {exact_dupes} 条 ({exact_dupes/len(data)*100:.1f}%)")
    if report.get("fuzzy_dupes", 0) > 0:
        warnings.append(f"⚠️ 近似重复 {report['fuzzy_dupes']} 条")
    if empty > 0:
        warnings.append(f"⚠️ 空答案 {empty} 条 ({empty/len(data)*100:.1f}%)")
    if len_stats["too_short"] > 0:
        warnings.append(f"⚠️ 过短(<10字) {len_stats['too_short']} 条")
    if report["repetitive_answers"]:
        top = report["repetitive_answers"][0]
        warnings.append(f"⚠️ 高频重复答案: \"{top['answer'][:50]}...\" 出现 {top['count']}次 ({top['ratio']}%)")
    if report["quality_score"] < 4:
        warnings.append(f"❌ 整体质量偏低 ({report['quality_score']}/10)")

    if len(data) < 50:
        suggestions.append("💡 数据量偏少(<50条)，建议增加到 200+ 条或开启数据增强")
    elif len(data) < 200:
        suggestions.append("💡 数据量较少(<200条)，建议增加训练轮数(3-5 epochs)")
    if exact_dupes + report.get("fuzzy_dupes", 0) > len(data) * 0.1:
        suggestions.append("🔧 重复率 >10%，建议点击「一键清洗」去重")
    if report["quality_score"] < 6:
        suggestions.append("🔧 质量偏低，建议检查数据源或使用做题学习模式生成高质量数据")
    if report["length_stats"]["avg"] > 10000:
        suggestions.append("💡 平均长度较大，建议将 max_seq_len 设为 4096+")
    if not warnings:
        suggestions.append("✅ 数据质量良好，可以直接训练！")

    report["warnings"] = warnings
    report["suggestions"] = suggestions

    return report


def clean_dataset(path: str,
                  do_dedup: bool = True,
                  do_fuzzy_dedup: bool = True,
                  do_filter_empty: bool = True,
                  do_filter_length: bool = True,
                  min_len: int = 10,
                  max_len: int = 50000,
                  fuzzy_threshold: float = 0.85) -> Dict[str, Any]:
    """一键清洗数据集

    Returns:
        {
            "original": int,
            "cleaned": int,
            "removed": {"exact_dedup": int, "fuzzy_dedup": int, "empty": int, "length": int},
            "output_path": str,
        }
    """
    data = _load_data(path)
    if not data:
        return {"original": 0, "cleaned": 0, "error": "无法加载数据"}

    original = len(data)
    removed = {"exact_dedup": 0, "fuzzy_dedup": 0, "empty": 0, "length": 0}

    if do_dedup:
        data, n = dedup_exact(data)
        removed["exact_dedup"] = n

    if do_fuzzy_dedup and len(data) <= 10000:
        data, n = dedup_fuzzy(data, fuzzy_threshold)
        removed["fuzzy_dedup"] = n

    if do_filter_empty:
        data, n = filter_empty_output(data)
        removed["empty"] = n

    if do_filter_length:
        data, stats = filter_by_length(data, min_len, max_len)
        removed["length"] = stats["too_short"] + stats["too_long"]

    # 保存清洗后的数据
    p = Path(path)
    if not p.exists():
        p = Path(DATASETS_DIR) / path
    if p.is_dir():
        out_name = f"{p.name}_cleaned.json"
    else:
        out_name = f"{p.stem}_cleaned.json"

    out_path = Path(DATASETS_DIR) / out_name
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    total_removed = sum(removed.values())
    log(f"数据清洗: {original} → {len(data)} (移除 {total_removed})")
    return {
        "original": original,
        "cleaned": len(data),
        "removed": removed,
        "total_removed": total_removed,
        "output_path": out_name,
    }


def format_report_markdown(report: Dict) -> str:
    """将体检报告格式化为 Markdown"""
    if report.get("error"):
        return f"❌ {report['error']}"

    lines = []
    total = report["total"]
    fmt = report.get("format", "unknown")
    fmt_names = {"alpaca": "指令对(Alpaca)", "sharegpt": "多轮对话(ShareGPT)",
                 "text": "纯文本", "dpo": "偏好对(DPO)", "mixed": "混合格式"}
    lines.append(f"### 📊 数据集体检报告")
    lines.append(f"**总量**: {total} 条 | **格式**: {fmt_names.get(fmt, fmt)}")
    lines.append("")

    # 质量评分
    score = report.get("quality_score", 0)
    grade = "🟢 优" if score >= 7 else "🟡 良" if score >= 5 else "🔴 差"
    lines.append(f"**质量评分**: {score}/10 {grade}")
    dist = report.get("score_distribution", {})
    if dist:
        dist_str = " | ".join(f"{k}: {v}" for k, v in dist.items())
        lines.append(f"分布: {dist_str}")
    lines.append("")

    # 长度
    ls = report.get("length_stats", {})
    if ls:
        lines.append(f"**长度**: 最短 {ls['min']} / 平均 {ls['avg']} / 中位 {ls['median']} / 最长 {ls['max']}")

    # 问题
    lines.append("")
    lines.append("**发现的问题**:")
    warnings = report.get("warnings", [])
    if warnings:
        for w in warnings:
            lines.append(f"  {w}")
    else:
        lines.append("  ✅ 未发现明显问题")

    # 建议
    lines.append("")
    suggestions = report.get("suggestions", [])
    if suggestions:
        lines.append("**优化建议**:")
        for s in suggestions:
            lines.append(f"  {s}")

    return "\n".join(lines)


# ════════════════════════════════════════════
#  数据增强（小数据集）
# ════════════════════════════════════════════

def augment_instructions(data: List[Dict], multiplier: int = 2) -> List[Dict]:
    """指令改写增强 — 不需要模型，用规则变换扩充数据

    策略:
    1. 同义替换（你/您，请/麻烦，解释/说明）
    2. 添加约束（加"用一句话"/"详细"前缀）
    3. 重排序（交换子句顺序）

    适用于 <500 条的小数据集。
    """
    import random

    # 同义替换对
    synonyms = [
        ("请", "麻烦"), ("解释", "说明"), ("描述", "介绍"),
        ("列举", "列出"), ("分析", "剖析"), ("总结", "概括"),
        ("如何", "怎样"), ("什么是", "什么叫"), ("为什么", "为何"),
        ("优点", "好处"), ("缺点", "不足"), ("区别", "差异"),
    ]

    # 约束前缀
    prefixes = [
        "请用简单的语言", "请详细地", "请简要地", "请举例说明",
        "从专业角度", "通俗易懂地",
    ]

    augmented = list(data)  # 保留原数据

    for _ in range(multiplier - 1):
        for item in data:
            inst = item.get("instruction", "")
            if not inst or len(inst) < 10:
                continue

            new_item = dict(item)
            strategy = random.choice(["synonym", "prefix", "both"])

            if strategy in ("synonym", "both"):
                # 随机替换 1-2 个同义词
                modified = inst
                replacements = random.sample(synonyms, min(2, len(synonyms)))
                for a, b in replacements:
                    if a in modified:
                        modified = modified.replace(a, b, 1)
                    elif b in modified:
                        modified = modified.replace(b, a, 1)
                new_item["instruction"] = modified

            if strategy in ("prefix", "both"):
                prefix = random.choice(prefixes)
                curr = new_item.get("instruction", inst)
                # 避免重复添加前缀
                if not any(curr.startswith(p) for p in prefixes):
                    new_item["instruction"] = f"{prefix}{curr}"

            # 只有真正改变了才加入
            if new_item["instruction"] != inst:
                augmented.append(new_item)

    return augmented


def augment_and_save(path: str, multiplier: int = 2) -> Dict[str, Any]:
    """增强数据集并保存

    Returns: {"original": int, "augmented": int, "output_path": str}
    """
    data = _load_data(path)
    if not data:
        return {"original": 0, "error": "无法加载数据"}

    augmented = augment_instructions(data, multiplier)

    p = Path(path)
    if not p.exists():
        p = Path(DATASETS_DIR) / path
    out_name = f"{p.stem}_augmented.json" if p.is_file() else f"{p.name}_augmented.json"
    out_path = Path(DATASETS_DIR) / out_name
    out_path.write_text(json.dumps(augmented, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    log(f"数据增强: {len(data)} → {len(augmented)} (×{multiplier})")
    return {
        "original": len(data),
        "augmented": len(augmented),
        "output_path": out_name,
    }
