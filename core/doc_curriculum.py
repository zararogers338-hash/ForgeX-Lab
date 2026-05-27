# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

"""ForgeX 文档课程训练引擎 — 把一本书教进模型参数

与 RAG / 知识库的本质区别:
  - RAG = 检索后拼接到 prompt（模型不真正"学会"）
  - 文档课程 = 把知识转化为训练数据，写入模型权重（模型真正"学会"）

核心流程:
  1. 文档解析   — 支持 PDF / TXT / MD / DOCX，自动识别结构
  2. 知识拆解   — 按章节/段落拆分为知识单元，估算难度
  3. QA 生成    — 每个知识单元生成多类型问答训练对
  4. 课程编排   — 从易到难递进（课程学习 Curriculum Learning）
  5. 分阶段训练 — 逐阶段 SFT，每阶段结束考试检验
  6. 学习报告   — 跟踪每阶段的掌握程度

QA 生成策略（6 种题型，覆盖布鲁姆认知层次）:
  L1 记忆   — "X是什么？"
  L2 理解   — "请用自己的话解释..."
  L3 应用   — "如何用X解决Y问题？"
  L4 分析   — "比较A和B的区别"
  L5 综合   — "基于以下多个概念，设计..."
  L6 评价   — "评估X方法的优缺点"

典型场景:
  - 上传一本技术手册 → 模型学会该领域专业知识
  - 上传公司文档 → 模型成为内部知识专家
  - 上传教材 → 模型成为该学科辅导老师
"""

from __future__ import annotations

import gc
import json
import math
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core import LORAS_DIR, DATASETS_DIR, log


# ════════════════════════════════════════════════════
#  1. 文档解析器
# ════════════════════════════════════════════════════

def parse_document(file_path: str) -> List[Dict[str, Any]]:
    """解析文档为结构化章节列表

    Returns: [{
        "title": "章节标题",
        "level": 1-3 (标题层级),
        "content": "正文内容",
        "page": 页码(如有),
    }, ...]
    """
    p = Path(file_path)
    suffix = p.suffix.lower()

    if suffix == ".txt":
        return _parse_txt(p)
    elif suffix == ".md":
        return _parse_markdown(p)
    elif suffix == ".pdf":
        return _parse_pdf(p)
    elif suffix in (".docx", ".doc"):
        return _parse_docx(p)
    elif suffix in (".jsonl", ".json"):
        return _parse_structured(p)
    else:
        # 尝试当纯文本读
        return _parse_txt(p)


def _parse_txt(p: Path) -> List[Dict]:
    """解析纯文本 — 按空行/长度拆段"""
    text = p.read_text(encoding="utf-8", errors="ignore")
    sections = []
    paragraphs = re.split(r'\n\s*\n', text)

    current_section = {"title": "正文", "level": 1, "content": "", "page": 0}
    chunk_chars = 0
    chunk_idx = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 检测标题模式
        is_title = False
        title_level = 1
        # 数字编号标题: "1. xxx", "第一章 xxx", "1.2 xxx"
        title_match = re.match(
            r'^(第[一二三四五六七八九十百]+[章节篇部]|'
            r'\d+[\.、）)]\s*|'
            r'[一二三四五六七八九十]+[、．.]\s*)'
            r'(.{2,50})$', para, re.MULTILINE
        )
        if title_match:
            is_title = True
            title_text = para
            if re.match(r'^第.+[章篇部]', para):
                title_level = 1
            elif re.match(r'^\d+\.', para):
                title_level = 2
            else:
                title_level = 2

        # 短行 + 无标点 → 可能是标题
        if not is_title and len(para) < 50 and not re.search(r'[。！？；]', para):
            is_title = True
            title_text = para
            title_level = 2

        if is_title:
            # 保存之前的段落
            if current_section["content"].strip():
                sections.append(dict(current_section))
            current_section = {
                "title": para[:60],
                "level": title_level,
                "content": "",
                "page": 0,
            }
            chunk_chars = 0
            chunk_idx += 1
        else:
            current_section["content"] += para + "\n\n"
            chunk_chars += len(para)

            # 过长自动分段（每 2000 字切一刀）
            if chunk_chars > 2000:
                sections.append(dict(current_section))
                chunk_idx += 1
                current_section = {
                    "title": f"{current_section['title']}（续）",
                    "level": current_section["level"],
                    "content": "",
                    "page": 0,
                }
                chunk_chars = 0

    if current_section["content"].strip():
        sections.append(current_section)

    # 如果只有一个巨大段落，按固定长度切
    if len(sections) <= 1 and sections and len(sections[0]["content"]) > 3000:
        return _split_long_section(sections[0])

    return sections if sections else [{"title": "文档内容", "level": 1,
                                        "content": text[:10000], "page": 0}]


def _parse_markdown(p: Path) -> List[Dict]:
    """解析 Markdown — 按标题层级拆分"""
    text = p.read_text(encoding="utf-8", errors="ignore")
    sections = []
    current = {"title": "导言", "level": 1, "content": "", "page": 0}

    for line in text.split("\n"):
        heading = re.match(r'^(#{1,4})\s+(.+)', line)
        if heading:
            if current["content"].strip():
                sections.append(dict(current))
            level = len(heading.group(1))
            current = {
                "title": heading.group(2).strip(),
                "level": min(level, 3),
                "content": "",
                "page": 0,
            }
        else:
            current["content"] += line + "\n"

    if current["content"].strip():
        sections.append(current)

    return sections if sections else [{"title": "文档", "level": 1,
                                        "content": text[:10000], "page": 0}]


def _parse_pdf(p: Path) -> List[Dict]:
    """解析 PDF"""
    text = ""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(p))
        page_texts = []
        for page_num, page in enumerate(doc):
            page_text = page.get_text("text")
            page_texts.append((page_num + 1, page_text))
        doc.close()

        # 合并所有页面文本，保留页码信息
        sections = []
        current = {"title": "正文", "level": 1, "content": "", "page": 1}
        chunk_chars = 0

        for page_num, page_text in page_texts:
            for para in page_text.split("\n\n"):
                para = para.strip()
                if not para:
                    continue

                # 检测标题（短行、大写开头、无标点结尾）
                if len(para) < 60 and not para.endswith(('。', '！', '？', '.', '!', '?')):
                    if current["content"].strip():
                        sections.append(dict(current))
                    current = {
                        "title": para[:60],
                        "level": 2,
                        "content": "",
                        "page": page_num,
                    }
                    chunk_chars = 0
                else:
                    current["content"] += para + "\n\n"
                    chunk_chars += len(para)

                    if chunk_chars > 2000:
                        sections.append(dict(current))
                        current = {
                            "title": f"{current['title']}（续）",
                            "level": current["level"],
                            "content": "",
                            "page": page_num,
                        }
                        chunk_chars = 0

        if current["content"].strip():
            sections.append(current)

        return sections if sections else [{"title": p.stem, "level": 1,
                                            "content": text[:10000], "page": 0}]
    except ImportError:
        # 回退: pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(str(p)) as pdf:
                text = "\n\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )
        except ImportError:
            raise ImportError(
                "需要安装 PDF 解析库: pip install PyMuPDF 或 pip install pdfplumber"
            )

    # 如果只拿到纯文本，当 txt 解析
    if text:
        tmp = Path("/tmp") / f"_forgex_pdf_{p.stem}.txt"
        tmp.write_text(text, encoding="utf-8")
        return _parse_txt(tmp)
    return [{"title": p.stem, "level": 1, "content": "（PDF 解析失败）", "page": 0}]


def _parse_docx(p: Path) -> List[Dict]:
    """解析 Word 文档"""
    try:
        from docx import Document
        doc = Document(str(p))
        sections = []
        current = {"title": "正文", "level": 1, "content": "", "page": 0}

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            # Word 标题样式检测
            style = (para.style.name or "").lower()
            if "heading" in style:
                if current["content"].strip():
                    sections.append(dict(current))
                level = 1
                if "2" in style:
                    level = 2
                elif "3" in style or "4" in style:
                    level = 3
                current = {"title": text[:60], "level": level, "content": "", "page": 0}
            else:
                current["content"] += text + "\n\n"

        if current["content"].strip():
            sections.append(current)

        return sections if sections else [{"title": p.stem, "level": 1,
                                            "content": "（DOCX 解析失败）", "page": 0}]
    except ImportError:
        raise ImportError("需要安装: pip install python-docx")


def _parse_structured(p: Path) -> List[Dict]:
    """解析已结构化的 JSON/JSONL"""
    sections = []
    if p.suffix == ".jsonl":
        for line in p.read_text(encoding="utf-8").strip().splitlines():
            try:
                obj = json.loads(line)
                text = obj.get("text", "") or obj.get("content", "") or json.dumps(obj, ensure_ascii=False)
                title = obj.get("title", "") or text[:40]
                sections.append({"title": title, "level": 2, "content": text, "page": 0})
            except Exception:
                continue
    else:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    text = item.get("text", "") or item.get("content", "") or json.dumps(item, ensure_ascii=False)
                    title = item.get("title", "") or f"条目 {i+1}"
                else:
                    text = str(item)
                    title = f"条目 {i+1}"
                sections.append({"title": title, "level": 2, "content": text, "page": 0})
    return sections


def _split_long_section(section: Dict, chunk_size: int = 1500) -> List[Dict]:
    """将过长的单个 section 切分为多段"""
    text = section["content"]
    parts = []
    sentences = re.split(r'(?<=[。！？.!?\n])\s*', text)
    current = ""
    idx = 0

    for sent in sentences:
        if len(current) + len(sent) > chunk_size and current:
            idx += 1
            parts.append({
                "title": f"{section['title']} (段{idx})",
                "level": section["level"],
                "content": current.strip(),
                "page": section.get("page", 0),
            })
            current = sent
        else:
            current += sent

    if current.strip():
        idx += 1
        parts.append({
            "title": f"{section['title']} (段{idx})" if idx > 0 else section["title"],
            "level": section["level"],
            "content": current.strip(),
            "page": section.get("page", 0),
        })
    return parts


# ════════════════════════════════════════════════════
#  2. 知识单元 + 难度估算
# ════════════════════════════════════════════════════

@dataclass
class KnowledgeUnit:
    """一个知识单元 = 可教学的最小知识块"""
    id: int
    title: str
    content: str
    level: int                    # 标题层级
    difficulty: float = 0.0       # 难度 0-1
    stage: int = 0                # 所属课程阶段
    key_concepts: List[str] = field(default_factory=list)
    char_count: int = 0

    def __post_init__(self):
        self.char_count = len(self.content)


def estimate_difficulty(text: str) -> float:
    """估算文本难度 (0-1)

    综合指标:
    - 平均句长（越长越难）
    - 专业词汇密度（英文/数字/术语占比越高越难）
    - 逻辑连接词密度（因此/但是/然而 越多越难）
    """
    if not text:
        return 0.0

    sentences = re.split(r'[。！？.!?\n]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if not sentences:
        return 0.3

    # 平均句长 (中文按字数)
    avg_len = sum(len(s) for s in sentences) / len(sentences)
    len_score = min(1.0, avg_len / 80)  # 80字/句 = max

    # 专业词汇密度 (英文单词 + 数字 + 特殊符号)
    total_chars = len(text)
    technical = len(re.findall(r'[a-zA-Z]+|[\d]+[.%]?|\$|#|@|\+|=', text))
    tech_score = min(1.0, technical / max(total_chars * 0.1, 1))

    # 逻辑连接词
    logic_words = len(re.findall(
        r'因此|所以|但是|然而|尽管|虽然|不过|此外|另外|'
        r'首先|其次|最后|总之|综上|换言之|也就是说|'
        r'therefore|however|moreover|furthermore|consequently',
        text, re.IGNORECASE
    ))
    logic_score = min(1.0, logic_words / max(len(sentences) * 0.3, 1))

    # 综合
    difficulty = len_score * 0.4 + tech_score * 0.35 + logic_score * 0.25
    return round(min(1.0, max(0.0, difficulty)), 3)


def extract_key_concepts(text: str, top_n: int = 5) -> List[str]:
    """提取关键概念（高频名词短语）"""
    # 中文: 2-6字的高频词组
    cn_phrases = re.findall(r'[\u4e00-\u9fff]{2,6}', text)
    # 英文: 大写开头或全大写的词
    en_phrases = re.findall(r'[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*|[A-Z]{2,}', text)

    all_phrases = cn_phrases + en_phrases
    counter = {}
    for p in all_phrases:
        if len(p) >= 2 and p not in ('这个', '那个', '我们', '他们', '可以', '通过',
                                       '进行', '使用', '其中', '以及', '或者', '如果'):
            counter[p] = counter.get(p, 0) + 1

    # 按频率排序
    sorted_concepts = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    return [c[0] for c in sorted_concepts[:top_n]]


def build_knowledge_units(sections: List[Dict]) -> List[KnowledgeUnit]:
    """将解析的章节转化为知识单元列表"""
    units = []
    for idx, sec in enumerate(sections):
        content = sec["content"].strip()
        if len(content) < 20:  # 过短跳过（降低阈值以保留更多内容）
            continue
        unit = KnowledgeUnit(
            id=idx,
            title=sec["title"],
            content=content,
            level=sec.get("level", 2),
            difficulty=estimate_difficulty(content),
            key_concepts=extract_key_concepts(content),
        )
        units.append(unit)

    # 兜底: 如果所有章节都太短，合并为一个大单元
    if not units and sections:
        merged = "\n\n".join(s["content"].strip() for s in sections if s["content"].strip())
        if merged:
            units.append(KnowledgeUnit(
                id=0,
                title=sections[0].get("title", "文档内容"),
                content=merged,
                level=1,
                difficulty=estimate_difficulty(merged),
                key_concepts=extract_key_concepts(merged),
            ))

    # 按难度排序并分配阶段
    units.sort(key=lambda u: u.difficulty)
    n = len(units)
    if n == 0:
        log("  知识拆解: 0 个单元（所有章节内容过短）")
        return units
    for i, u in enumerate(units):
        if n <= 3:
            u.stage = 1
        else:
            u.stage = min(3, int(i / n * 3) + 1)  # 1-3 阶段

    log(f"  知识拆解: {len(units)} 个单元, "
        f"难度 {min(u.difficulty for u in units):.2f}-{max(u.difficulty for u in units):.2f}")
    return units


# ════════════════════════════════════════════════════
#  3. QA 生成器 — 抗幻觉设计 (extractive-first)
# ════════════════════════════════════════════════════

# 核心原则: 所有回答必须可溯源到文档原文
# - 模板 QA: 答案直接从原文中提取相关句子组装
# - 模型 QA: 严格限定"只能使用原文信息"，生成后验证

# 不同阶段使用的题型权重
STAGE_WEIGHTS = {
    1: {"recall": 5, "understand": 3, "apply": 1, "analyze": 0, "synthesize": 0, "evaluate": 0},
    2: {"recall": 2, "understand": 3, "apply": 3, "analyze": 2, "synthesize": 0, "evaluate": 0},
    3: {"recall": 1, "understand": 1, "apply": 2, "analyze": 3, "synthesize": 2, "evaluate": 2},
}


def _extract_relevant_sentences(content: str, concept: str, max_sentences: int = 5) -> str:
    """从原文中提取与 concept 相关的句子（抗幻觉核心）"""
    sentences = re.split(r'(?<=[。！？.!?\n])\s*', content)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if not sentences:
        return content[:500]

    concept_words = set(re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]+', concept.lower()))

    # 按与 concept 的相关度排序
    scored = []
    for s in sentences:
        s_words = set(re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]+', s.lower()))
        overlap = len(concept_words & s_words)
        # 直接包含 concept 全名 → 最高分
        direct = 2 if concept.lower() in s.lower() else 0
        scored.append((overlap + direct, s))

    # 取最相关的句子，保持原文顺序
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:max_sentences]
    # 按在原文中的位置重排
    top_set = {s for _, s in top}
    result = [s for s in sentences if s in top_set]
    return " ".join(result) if result else sentences[0]


def _extract_all_facts(content: str, max_len: int = 600) -> str:
    """提取原文核心事实句（去除过渡句）"""
    sentences = re.split(r'(?<=[。！？.!?\n])\s*', content)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    # 过滤过渡句/空洞句
    fact_sentences = []
    skip_patterns = re.compile(r'^(总之|综上|此外|另外|最后|首先|其次|接下来|下面)')
    for s in sentences:
        if skip_patterns.match(s):
            continue
        fact_sentences.append(s)

    result = ""
    for s in fact_sentences:
        if len(result) + len(s) > max_len:
            break
        result += s + " "
    return result.strip() or content[:max_len]


def _check_hallucination(answer: str, source: str, threshold: float = 0.25) -> float:
    """检查回答与原文的关键词重叠率，返回 0-1 忠实度分数

    threshold: 低于此值判定为幻觉
    """
    if not answer or not source:
        return 0.0
    ans_words = set(re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}', answer.lower()))
    src_words = set(re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}', source.lower()))
    if not ans_words:
        return 0.0
    overlap = len(ans_words & src_words)
    return overlap / len(ans_words)


# 模板: 问题 → (问题模板, 回答构造函数)
# 回答从原文提取，不生成新内容

def _build_recall_qa(concept: str, content: str) -> Dict:
    relevant = _extract_relevant_sentences(content, concept, max_sentences=4)
    return {
        "instruction": f"请解释「{concept}」是什么。",
        "input": "",
        "output": f"{concept}的含义如下：{relevant}",
    }


def _build_understand_qa(concept: str, content: str) -> Dict:
    relevant = _extract_relevant_sentences(content, concept, max_sentences=3)
    return {
        "instruction": f"请用简洁的语言说明「{concept}」的核心含义。",
        "input": "",
        "output": f"关于{concept}，核心要点是：{relevant}",
    }


def _build_apply_qa(concept: str, content: str) -> Dict:
    relevant = _extract_relevant_sentences(content, concept, max_sentences=4)
    return {
        "instruction": f"如何在实际场景中应用「{concept}」？",
        "input": "",
        "output": f"应用{concept}的方法：根据文档所述，{relevant}",
    }


def _build_analyze_qa(concept: str, content: str) -> Dict:
    facts = _extract_all_facts(content, max_len=500)
    return {
        "instruction": f"分析「{concept}」的特点和要点。",
        "input": "",
        "output": f"对{concept}的分析：{facts}",
    }


def _build_synthesize_qa(concept: str, content: str) -> Dict:
    facts = _extract_all_facts(content, max_len=500)
    return {
        "instruction": f"综合你对「{concept}」的知识，总结关键信息。",
        "input": "",
        "output": f"综合来看，{concept}的关键信息包括：{facts}",
    }


def _build_evaluate_qa(concept: str, content: str) -> Dict:
    relevant = _extract_relevant_sentences(content, concept, max_sentences=4)
    return {
        "instruction": f"评价「{concept}」的价值和适用性。",
        "input": "",
        "output": f"关于{concept}的评价：根据所学内容，{relevant}",
    }


# 问句多样性模板（同一个 builder 配多种问法）
_QUESTION_VARIANTS = {
    "recall": [
        "请解释「{concept}」是什么。",
        "「{concept}」的定义和核心内容是什么？",
        "请说明「{concept}」的基本概念。",
        "什么是「{concept}」？请详细说明。",
    ],
    "understand": [
        "请用简洁的语言说明「{concept}」的核心含义。",
        "如果要向新手解释「{concept}」，你会怎么说？",
        "请用自己的话概括「{concept}」的要点。",
    ],
    "apply": [
        "如何在实际场景中应用「{concept}」？",
        "请举例说明「{concept}」的实际用途。",
        "在工作中遇到「{concept}」相关问题该怎么处理？",
    ],
    "analyze": [
        "分析「{concept}」的特点和要点。",
        "「{concept}」有哪些优势和局限？",
        "请比较分析「{concept}」的不同方面。",
    ],
    "synthesize": [
        "综合你对「{concept}」的知识，总结关键信息。",
        "请系统梳理「{concept}」的核心知识点。",
    ],
    "evaluate": [
        "评价「{concept}」的价值和适用性。",
        "如何评估「{concept}」的实际效果？",
    ],
}

QA_BUILDERS = {
    "recall": _build_recall_qa,
    "understand": _build_understand_qa,
    "apply": _build_apply_qa,
    "analyze": _build_analyze_qa,
    "synthesize": _build_synthesize_qa,
    "evaluate": _build_evaluate_qa,
}


def generate_qa_template(
    unit: KnowledgeUnit,
    qa_per_unit: int = 5,
    stage: int = 1,
) -> List[Dict[str, str]]:
    """用 extractive 模板为知识单元生成 QA 训练对

    核心: 答案全部从原文提取，零幻觉
    """
    qa_pairs = []
    concepts = unit.key_concepts or [unit.title]
    weights = STAGE_WEIGHTS.get(stage, STAGE_WEIGHTS[1])

    # 按权重选择题型
    types = []
    for t, w in weights.items():
        types.extend([t] * w)
    if not types:
        types = ["recall", "understand"]

    used_questions = set()  # 避免重复问题

    for _ in range(qa_per_unit):
        qa_type = random.choice(types)
        concept = random.choice(concepts)
        builder = QA_BUILDERS.get(qa_type, _build_recall_qa)

        qa = builder(concept, unit.content)

        # 用变体问句增加多样性
        variants = _QUESTION_VARIANTS.get(qa_type, [])
        if variants:
            q = random.choice(variants).format(concept=concept)
            if q not in used_questions:
                qa["instruction"] = q
                used_questions.add(q)

        qa["metadata"] = {
            "unit_id": unit.id,
            "unit_title": unit.title,
            "qa_type": qa_type,
            "stage": stage,
            "source": "extractive_template",
        }
        qa_pairs.append(qa)

    return qa_pairs


def generate_qa_with_model(
    unit: KnowledgeUnit,
    chat_fn: Callable[[str], str],
    qa_per_unit: int = 5,
    stage: int = 1,
) -> List[Dict[str, str]]:
    """用模型为知识单元生成 QA 训练对 — 带幻觉检测

    策略:
    1. 严格限定 prompt: "只能使用原文信息，不得添加原文未提及的内容"
    2. 生成后用 _check_hallucination 验证
    3. 未通过验证的 QA → 替换为 extractive 模板 QA
    """
    weights = STAGE_WEIGHTS.get(stage, STAGE_WEIGHTS[1])
    active_types = [t for t, w in weights.items() if w > 0]

    type_names_cn = {
        "recall": "记忆/复述", "understand": "理解/解释",
        "apply": "应用/实践", "analyze": "分析/比较",
        "synthesize": "综合/设计", "evaluate": "评价/评估",
    }

    # 严格 grounding prompt — 核心抗幻觉设计
    prompt = (
        f"你是一个教学专家。请严格根据以下【原文内容】生成 {qa_per_unit} 个问答训练对。\n\n"
        f"【原文内容】\n{unit.content[:1800]}\n\n"
        f"【严格要求 — 务必遵守】\n"
        f"1. 回答中的每一句话都必须能在原文中找到对应依据\n"
        f"2. 绝对不要添加原文中没有提到的信息、数据或例子\n"
        f"3. 如果原文信息不足以回答某个问题，就不要生成该问题\n"
        f"4. 回答应该直接引用或紧贴原文表述\n"
        f"5. 题型分布: {', '.join(type_names_cn.get(t, t) for t in active_types)}\n\n"
        f"格式:\n"
        f"Q: <问题>\n"
        f"A: <回答（必须基于原文）>\n"
        f"---\n\n"
        f"请生成:"
    )

    try:
        result = chat_fn(prompt)
    except Exception as e:
        log(f"  ⚠️ 模型 QA 生成失败，回退模板: {e}")
        return generate_qa_template(unit, qa_per_unit, stage)

    # 解析模型输出
    qa_pairs = []
    blocks = re.split(r'---+|\n\n(?=Q[:：])', result)

    for block in blocks:
        q_match = re.search(r'Q[:：]\s*(.+?)(?=\nA[:：])', block, re.DOTALL)
        a_match = re.search(r'A[:：]\s*(.+)', block, re.DOTALL)
        if q_match and a_match:
            q = q_match.group(1).strip()
            a = a_match.group(1).strip()
            if len(q) > 10 and len(a) > 20:
                # 幻觉检测: 检查回答与原文的关键词重叠率
                fidelity = _check_hallucination(a, unit.content)
                if fidelity >= 0.25:  # 至少 25% 关键词来自原文
                    qa_pairs.append({
                        "instruction": q,
                        "input": "",
                        "output": a,
                        "metadata": {
                            "unit_id": unit.id,
                            "unit_title": unit.title,
                            "qa_type": "model_generated",
                            "stage": stage,
                            "source": "model_verified",
                            "fidelity": round(fidelity, 3),
                        },
                    })
                else:
                    log(f"    ⚠️ 幻觉过滤: fidelity={fidelity:.2f} < 0.25, Q={q[:30]}...")

    # 不够就用 extractive 模板补齐
    if len(qa_pairs) < qa_per_unit:
        extras = generate_qa_template(unit, qa_per_unit - len(qa_pairs), stage)
        qa_pairs.extend(extras)

    return qa_pairs[:qa_per_unit]

    # 不够就用模板补
    if len(qa_pairs) < qa_per_unit:
        extras = generate_qa_template(unit, qa_per_unit - len(qa_pairs), stage)
        qa_pairs.extend(extras)

    return qa_pairs[:qa_per_unit]


def generate_exam(
    units: List[KnowledgeUnit],
    chat_fn: Optional[Callable] = None,
    num_questions: int = 10,
) -> List[Dict[str, str]]:
    """为一组知识单元生成考试题"""
    questions = []
    for unit in random.sample(units, min(len(units), num_questions)):
        concept = random.choice(unit.key_concepts) if unit.key_concepts else unit.title
        excerpt = unit.content[:300]

        questions.append({
            "question": f"请简要回答: {concept}的核心要点是什么？",
            "reference": _extract_all_facts(unit.content, 300),
            "unit_id": unit.id,
            "unit_title": unit.title,
        })
    return questions


def grade_exam(
    questions: List[Dict],
    chat_fn: Callable[[str], str],
) -> Tuple[float, List[Dict]]:
    """用模型回答考题并自评分数

    Returns: (average_score, detailed_results)
    """
    results = []
    for q in questions:
        try:
            answer = chat_fn(q["question"]).strip()
        except Exception:
            answer = ""

        # 简单关键词重叠评分
        ref_words = set(re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]+', q["reference"].lower()))
        ans_words = set(re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]+', answer.lower()))

        if not ref_words:
            score = 0.5
        else:
            overlap = len(ref_words & ans_words)
            score = min(1.0, overlap / max(len(ref_words) * 0.3, 1))

        results.append({
            "question": q["question"],
            "answer": answer[:200],
            "reference_excerpt": q["reference"][:100],
            "score": round(score, 2),
            "unit_title": q["unit_title"],
        })

    avg = sum(r["score"] for r in results) / max(len(results), 1)
    return round(avg, 3), results


# ════════════════════════════════════════════════════
#  4. 课程编排器
# ════════════════════════════════════════════════════

@dataclass
class CurriculumStage:
    """一个课程阶段"""
    stage_num: int
    title: str
    units: List[KnowledgeUnit]
    qa_data: List[Dict] = field(default_factory=list)
    exam_score: float = 0.0


def plan_curriculum(
    units: List[KnowledgeUnit],
    num_stages: int = 3,
) -> List[CurriculumStage]:
    """将知识单元编排为递进课程

    阶段划分策略:
    - Stage 1 (基础): 低难度 + 记忆/理解题为主
    - Stage 2 (进阶): 中难度 + 应用/分析题为主
    - Stage 3 (精通): 高难度 + 综合/评价题为主
    """
    # 按难度排序
    sorted_units = sorted(units, key=lambda u: u.difficulty)
    n = len(sorted_units)
    num_stages = min(num_stages, max(1, n // 2))  # 至少每阶段 2 个单元

    stages = []
    stage_names = {
        1: "📗 基础篇 — 概念入门",
        2: "📘 进阶篇 — 深入理解",
        3: "📕 精通篇 — 融会贯通",
    }

    for s in range(num_stages):
        start = int(s / num_stages * n)
        end = int((s + 1) / num_stages * n)
        stage_units = sorted_units[start:end]
        for u in stage_units:
            u.stage = s + 1

        stages.append(CurriculumStage(
            stage_num=s + 1,
            title=stage_names.get(s + 1, f"阶段 {s+1}"),
            units=stage_units,
        ))

    log(f"  课程编排: {num_stages} 阶段, "
        + ", ".join(f"S{s.stage_num}={len(s.units)}单元" for s in stages))
    return stages


# ════════════════════════════════════════════════════
#  5. 主引擎: 文档课程训练
# ════════════════════════════════════════════════════

def doc_curriculum_train(
    model_path: str,
    doc_paths: List[str],
    output_name: str = "doc_trained",
    num_stages: int = 3,
    qa_per_unit: int = 5,
    use_model_qa: bool = False,
    lr: float = 2e-4,
    epochs_per_stage: float = 2.0,
    batch_size: int = 1,
    max_seq: int = 2048,
    lora_rank: int = 64,
    task=None,
) -> Dict[str, Any]:
    """文档课程训练主流程

    Args:
        model_path: 基座模型路径
        doc_paths: 文档文件路径列表
        output_name: 输出模型名
        num_stages: 课程阶段数 (1-3)
        qa_per_unit: 每个知识单元生成多少 QA
        use_model_qa: 是否用模型生成 QA（质量更高，更慢）
        lr, epochs_per_stage, ...: 训练参数

    Returns: 训练报告
    """
    from core.trainer import TrainerEngine

    _upd = lambda p, m: (task.update_progress(p, m) if task else log(m))
    start_time = time.time()

    results = {
        "output_dir": "",
        "stages_completed": 0,
        "total_qa_samples": 0,
        "documents_parsed": 0,
        "knowledge_units": 0,
        "stage_reports": [],
        "exam_scores": [],
    }

    output_base = Path(DATASETS_DIR) / f"_curriculum_{output_name}"
    output_base.mkdir(parents=True, exist_ok=True)

    # ── Step 1: 解析所有文档 ──
    _upd(2, f"Step 1: 解析 {len(doc_paths)} 个文档...")
    all_sections = []
    for doc_path in doc_paths:
        try:
            sections = parse_document(doc_path)
            all_sections.extend(sections)
            log(f"  📄 {Path(doc_path).name}: {len(sections)} 个章节")
            results["documents_parsed"] += 1
        except Exception as e:
            log(f"  ❌ 解析失败 {doc_path}: {e}")

    if not all_sections:
        raise ValueError("所有文档解析失败，无法继续")

    # ── Step 2: 构建知识单元 ──
    _upd(8, "Step 2: 构建知识单元...")
    units = build_knowledge_units(all_sections)
    results["knowledge_units"] = len(units)
    log(f"  知识单元: {len(units)} 个")

    if not units:
        raise ValueError("未能提取有效知识单元")

    # ── Step 3: 课程编排 ──
    _upd(10, "Step 3: 课程编排...")
    stages = plan_curriculum(units, num_stages)

    # ── Step 4: 生成 QA + 分阶段训练 ──
    current_model = model_path
    chat_fn = None

    for stage in stages:
        stage_pct_base = 10 + int(stage.stage_num / len(stages) * 80)
        _upd(stage_pct_base, f"═══ {stage.title} ({len(stage.units)} 单元) ═══")
        log(f"\n{'='*50}")
        log(f"  📚 {stage.title}")
        log(f"{'='*50}")

        # 4a. 生成 QA 数据
        _upd(stage_pct_base + 2, f"S{stage.stage_num}: 生成 QA 训练数据...")

        if use_model_qa and chat_fn is None:
            try:
                from core.self_evolve import _build_local_chat_fn
                chat_fn = _build_local_chat_fn(current_model, temperature=0.3, max_new_tokens=800)
            except Exception as e:
                log(f"  ⚠️ 加载模型失败，使用模板 QA: {e}")
                use_model_qa = False

        all_qa = []
        for u_idx, unit in enumerate(stage.units):
            if use_model_qa and chat_fn:
                qa = generate_qa_with_model(unit, chat_fn, qa_per_unit, stage.stage_num)
            else:
                qa = generate_qa_template(unit, qa_per_unit, stage.stage_num)
            all_qa.extend(qa)

            if task and u_idx % 3 == 0:
                _upd(stage_pct_base + 5,
                     f"S{stage.stage_num}: QA 生成 {u_idx+1}/{len(stage.units)}")

        stage.qa_data = all_qa
        results["total_qa_samples"] += len(all_qa)
        log(f"  QA 数据: {len(all_qa)} 条")

        # 4b. 保存训练数据
        stage_data_path = output_base / f"stage_{stage.stage_num}_train.jsonl"
        with open(stage_data_path, "w", encoding="utf-8") as f:
            for item in all_qa:
                # 去掉 metadata，只保留训练字段
                train_item = {k: v for k, v in item.items() if k != "metadata"}
                f.write(json.dumps(train_item, ensure_ascii=False) + "\n")
        log(f"  训练集: {stage_data_path}")

        # 释放推理模型显存（训练需要显存）
        if chat_fn:
            try:
                from core.self_evolve import _cleanup_chat_fn
                _cleanup_chat_fn(chat_fn)
                chat_fn = None
            except Exception:
                pass

        # 4c. 训练
        train_pct = stage_pct_base + int(80 / len(stages) * 0.5)
        _upd(train_pct, f"S{stage.stage_num}: 训练模型...")

        stage_output = f"{output_name}_stage{stage.stage_num}"

        try:
            trainer = TrainerEngine()
            train_params = {
                "output_name": stage_output,
                "lr": lr,
                "batch_size": batch_size,
                "epochs": epochs_per_stage,
                "max_seq_len": max_seq,
                "use_qlora": True,
                "rank": lora_rank,
            }
            trainer.train(
                method="sft",
                backend="auto",
                base_model=current_model,
                dataset_path=str(stage_data_path),
                params=train_params,
                task=task,
            )
            current_model = str(LORAS_DIR / stage_output)
            log(f"  ✅ 训练完成: {stage_output}")
        except Exception as e:
            log(f"  ❌ S{stage.stage_num} 训练失败: {e}")
            results["stage_reports"].append({
                "stage": stage.stage_num,
                "title": stage.title,
                "error": str(e),
            })
            continue

        # 4d. 阶段考试
        exam_pct = stage_pct_base + int(80 / len(stages) * 0.85)
        _upd(exam_pct, f"S{stage.stage_num}: 阶段考试...")

        try:
            from core.self_evolve import _build_local_chat_fn, _cleanup_chat_fn
            exam_fn = _build_local_chat_fn(current_model, temperature=0.1, max_new_tokens=500)
            exam_questions = generate_exam(stage.units, num_questions=min(10, len(stage.units)))
            avg_score, exam_details = grade_exam(exam_questions, exam_fn)
            stage.exam_score = avg_score
            results["exam_scores"].append(avg_score)
            _cleanup_chat_fn(exam_fn)
            log(f"  📝 考试得分: {avg_score:.1%}")
        except Exception as e:
            log(f"  ⚠️ 考试失败: {e}")
            avg_score = 0.0
            exam_details = []

        results["stages_completed"] = stage.stage_num
        results["stage_reports"].append({
            "stage": stage.stage_num,
            "title": stage.title,
            "units": len(stage.units),
            "qa_samples": len(all_qa),
            "exam_score": avg_score,
            "output_dir": current_model,
        })

    # ── 最终结果 ──
    results["output_dir"] = current_model
    total_time = time.time() - start_time

    _upd(100, f"✅ 课程完成! {results['stages_completed']} 阶段, "
         f"{results['total_qa_samples']} QA, "
         f"耗时 {total_time/60:.1f}min")

    # 保存学习报告
    report = {
        **results,
        "total_time_minutes": round(total_time / 60, 1),
        "documents": [str(p) for p in doc_paths],
        "config": {
            "num_stages": num_stages, "qa_per_unit": qa_per_unit,
            "use_model_qa": use_model_qa, "lr": lr,
            "epochs_per_stage": epochs_per_stage,
        },
    }
    report_path = output_base / "curriculum_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    log(f"  📊 学习报告: {report_path}")

    return results


# ════════════════════════════════════════════════════
#  6. UI 提交入口
# ════════════════════════════════════════════════════

def doc_curriculum_submit(
    model_path: str,
    doc_files: List,
    output_name: str,
    num_stages: int = 3,
    qa_per_unit: int = 5,
    use_model_qa: bool = False,
    lr: float = 2e-4,
    epochs: float = 2.0,
    max_seq: int = 2048,
    rank: int = 64,
    task=None,
) -> Dict[str, Any]:
    """从 UI 提交文档课程训练"""
    # 处理文件路径
    paths = []
    for f in (doc_files or []):
        if hasattr(f, 'name'):
            paths.append(f.name)
        elif isinstance(f, str):
            paths.append(f)
    if not paths:
        raise ValueError("请上传至少一个文档")

    return doc_curriculum_train(
        model_path=model_path,
        doc_paths=paths,
        output_name=output_name or f"doc_{int(time.time())}",
        num_stages=int(num_stages),
        qa_per_unit=int(qa_per_unit),
        use_model_qa=bool(use_model_qa),
        lr=float(lr),
        epochs_per_stage=float(epochs),
        max_seq=int(max_seq),
        lora_rank=int(rank),
        task=task,
    )


def preview_document(file_path: str) -> Dict[str, Any]:
    """预览文档结构（不训练，只分析）"""
    sections = parse_document(file_path)
    units = build_knowledge_units(sections)
    stages = plan_curriculum(units)

    return {
        "file": Path(file_path).name,
        "sections": len(sections),
        "units": len(units),
        "stages": len(stages),
        "stage_detail": [
            {
                "stage": s.stage_num,
                "title": s.title,
                "units": len(s.units),
                "avg_difficulty": round(
                    sum(u.difficulty for u in s.units) / max(len(s.units), 1), 2
                ),
                "unit_titles": [u.title for u in s.units[:5]],
            }
            for s in stages
        ],
        "total_chars": sum(u.char_count for u in units),
        "key_concepts": list(set(
            c for u in units for c in u.key_concepts[:3]
        ))[:20],
        "difficulty_range": f"{min(u.difficulty for u in units):.2f} - {max(u.difficulty for u in units):.2f}" if units else "N/A",
    }
