# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

"""ForgeX 模型编辑器 — 给模型加系统提示词、知识库、身份卡、推理参数

把训练好的模型从"裸机"变成"开箱即用的产品":
  - 系统提示词:  定义模型人设和行为规则
  - 知识库:      嵌入参考文档，模型回答时自动检索
  - 身份卡:      名称、描述、标签、用途
  - 推理参数:    temperature / top_p / 重复惩罚 等默认值
  - Chat 模板:   自定义对话格式（Jinja2）
  - Modelfile:   一键生成 Ollama Modelfile 全量配置
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core import LORAS_DIR, log

# 统一配置文件名
META_FILE = "forgex_model_profile.json"


def _model_dir(name_or_path: str) -> Path:
    """解析模型目录"""
    p = Path(name_or_path)
    if p.is_dir():
        return p
    p = LORAS_DIR / name_or_path
    if p.is_dir():
        return p
    raise FileNotFoundError(f"找不到模型: {name_or_path}")


# ════════════════════════════════════════════════════
#  模型档案 (Model Profile)
# ════════════════════════════════════════════════════

def _auto_populate_from_hf(model_dir: Path, profile: Dict[str, Any]):
    """从 HuggingFace 标准文件补充 profile 中的空字段。
    
    只填充值为空/默认的字段，不覆盖用户已编辑的内容。
    读取: config.json, generation_config.json, tokenizer_config.json, README.md
    """
    # ── 1. config.json → 名称、描述、标签 ──
    cfg_path = model_dir / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            mt = cfg.get("model_type", "")
            archs = cfg.get("architectures", [])
            if mt and not profile.get("tags"):
                profile["tags"] = [mt]
            name_or_path = cfg.get("_name_or_path", "")
            if name_or_path and name_or_path != str(model_dir):
                if not profile.get("description"):
                    profile["description"] = f"基于 {name_or_path}"
                if not profile.get("name") or profile["name"] == model_dir.name:
                    profile["name"] = name_or_path.split("/")[-1] if "/" in name_or_path else model_dir.name
            if archs and not profile.get("tags"):
                tags = []
                for a in archs:
                    tag = a.replace("ForCausalLM", "").replace("LMHeadModel", "")
                    if tag:
                        tags.append(tag)
                if tags:
                    profile["tags"] = tags
        except Exception:
            pass

    # ── 2. generation_config.json → 推理参数(仅填充默认值的字段) ──
    gen_cfg_path = model_dir / "generation_config.json"
    if gen_cfg_path.exists():
        try:
            gc = json.loads(gen_cfg_path.read_text(encoding="utf-8"))
            params = profile.get("parameters", {})
            _defaults = {"temperature": 0.7, "top_p": 0.9, "top_k": 50,
                         "repeat_penalty": 1.1, "max_tokens": 2048}
            _gc_map = {
                "temperature": ("temperature", float),
                "top_p": ("top_p", float),
                "top_k": ("top_k", int),
                "repetition_penalty": ("repeat_penalty", float),
                "max_new_tokens": ("max_tokens", int),
            }
            for gc_key, (param_key, conv) in _gc_map.items():
                if gc_key in gc and gc[gc_key] is not None:
                    # 仅当 param 仍为默认值时才覆盖
                    if params.get(param_key) == _defaults.get(param_key):
                        params[param_key] = conv(gc[gc_key])
            if "max_length" in gc and gc["max_length"] and params.get("max_tokens") == 2048:
                params["max_tokens"] = int(gc["max_length"])
            profile["parameters"] = params
        except Exception:
            pass

    # ── 3. tokenizer_config.json → chat_template, 停止序列 ──
    tok_cfg_path = model_dir / "tokenizer_config.json"
    if tok_cfg_path.exists():
        try:
            tc = json.loads(tok_cfg_path.read_text(encoding="utf-8"))
            ct = tc.get("chat_template", "")
            if ct and not profile.get("chat_template"):
                profile["chat_template"] = ct
            eos = tc.get("eos_token")
            if isinstance(eos, dict):
                eos = eos.get("content", "")
            if eos and isinstance(eos, str):
                seqs = profile.get("parameters", {}).get("stop_sequences", [])
                if not seqs and eos:
                    profile.setdefault("parameters", {})["stop_sequences"] = [eos]
        except Exception:
            pass

    # ── 4. README.md → 描述 ──
    readme_path = model_dir / "README.md"
    if readme_path.exists() and not profile.get("description"):
        try:
            text = readme_path.read_text(encoding="utf-8", errors="ignore")
            if text.startswith("---"):
                end = text.find("---", 3)
                if end > 0:
                    text = text[end + 3:]
            lines = [l.strip() for l in text.split("\n")
                     if l.strip() and not l.strip().startswith("#") and not l.strip().startswith("|")]
            if lines:
                desc = " ".join(lines[:3])[:200]
                if desc:
                    profile["description"] = desc
        except Exception:
            pass

    # ── 5. adapter_config.json → LoRA 信息 ──
    adapter_path = model_dir / "adapter_config.json"
    if adapter_path.exists():
        try:
            ac = json.loads(adapter_path.read_text(encoding="utf-8"))
            base = ac.get("base_model_name_or_path", "")
            if base:
                if not profile.get("description"):
                    profile["description"] = f"LoRA 适配器, 基座: {base}"
                tags = profile.get("tags", [])
                if "LoRA" not in tags:
                    tags.append("LoRA")
                    profile["tags"] = tags
                profile["_base_model"] = base
        except Exception:
            pass


def load_profile(model_path: str) -> Dict[str, Any]:


    """加载模型档案（不存在则返回默认值）"""
    d = _model_dir(model_path)
    meta_path = d / META_FILE
    profile = {
        "name": d.name,
        "description": "",
        "author": "",
        "tags": [],
        "use_case": "",
        "system_prompt": "",
        "knowledge_docs": [],
        "chat_template": "",
        "parameters": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repeat_penalty": 1.1,
            "max_tokens": 2048,
            "stop_sequences": [],
        },
        "safety": {
            "enabled": False,
            "refusal_topics": [],
            "content_filter": "",
            "require_disclaimer": False,
            "disclaimer_text": "",
        },
        "output_format": {
            "mode": "free",
            "json_schema": "",
            "custom_template": "",
            "language": "",
            "max_paragraphs": 0,
        },
        "persona_presets": [],
        "active_preset": "",
        "created_at": "",
        "updated_at": "",
    }
    if meta_path.exists():
        try:
            saved = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                profile.update(saved)
        except Exception:
            pass
    # 无论 forge_profile 是否存在，都从 HF 文件补充空字段
    _auto_populate_from_hf(d, profile)

    # 兼容: 旧式 system_prompt.txt
    sp_file = d / "system_prompt.txt"
    if sp_file.exists() and not profile.get("system_prompt"):
        try:
            profile["system_prompt"] = sp_file.read_text(encoding="utf-8").strip()
        except Exception:
            pass

    # 读取模型架构信息
    cfg_path = d / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            profile["_architecture"] = {
                "model_type": cfg.get("model_type", "unknown"),
                "hidden_size": cfg.get("hidden_size", 0),
                "num_hidden_layers": cfg.get("num_hidden_layers", 0),
                "num_attention_heads": cfg.get("num_attention_heads", 0),
                "intermediate_size": cfg.get("intermediate_size", 0),
                "vocab_size": cfg.get("vocab_size", 0),
                "max_position_embeddings": cfg.get("max_position_embeddings", 0),
            }
        except Exception:
            pass

    # 估算参数量
    try:
        import os
        total_size = sum(
            f.stat().st_size for f in d.iterdir()
            if f.suffix in (".safetensors", ".bin")
        )
        profile["_estimated_params"] = f"{total_size / 2 / 1e9:.2f}B"
    except Exception:
        pass

    return profile


def save_profile(model_path: str, profile: Dict[str, Any]) -> str:
    """保存模型档案（完全防递归）"""
    d = _model_dir(model_path)
    meta_path = d / META_FILE

    # 清理内部字段
    clean = {}
    for k, v in profile.items():
        if isinstance(k, str) and not k.startswith("_"):
            clean[k] = v
    clean["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if not clean.get("created_at"):
        clean["created_at"] = clean["updated_at"]

    # 防递归 default 函数 — 追踪已见对象，拒绝展开 Gradio 组件
    _seen = set()

    def _safe_default(o):
        oid = id(o)
        if oid in _seen:
            return "<circular>"
        _seen.add(oid)
        try:
            s = str(o)
            return s[:500] if len(s) > 500 else s
        except Exception:
            return repr(type(o))

    # 安全序列化
    try:
        json_str = json.dumps(clean, indent=2, ensure_ascii=False, default=_safe_default)
    except (RecursionError, ValueError, TypeError):
        # 核弹级后备: 逐键单独序列化
        safe = {}
        for k, v in clean.items():
            try:
                json.dumps(v, default=str)
                safe[k] = v
            except Exception:
                try:
                    safe[k] = str(v)[:500]
                except Exception:
                    safe[k] = f"<unserializable:{type(v).__name__}>"
        json_str = json.dumps(safe, indent=2, ensure_ascii=False, default=str)

    meta_path.write_text(json_str, encoding="utf-8")

    # 同时写 system_prompt.txt
    sp = clean.get("system_prompt", "")
    if sp and isinstance(sp, str):
        (d / "system_prompt.txt").write_text(sp, encoding="utf-8")

    log(f"✅ 模型档案已保存: {d.name}")
    return str(meta_path)


# ════════════════════════════════════════════════════
#  知识库管理
# ════════════════════════════════════════════════════

def add_knowledge_doc(model_path: str, doc_path: str, doc_name: str = "") -> Dict:
    """添加知识文档到模型"""
    d = _model_dir(model_path)
    kb_dir = d / "knowledge_base"
    kb_dir.mkdir(exist_ok=True)

    src = Path(doc_path)
    if not src.exists():
        raise FileNotFoundError(f"文档不存在: {doc_path}")

    # 复制到知识库
    dst_name = doc_name or src.name
    dst = kb_dir / dst_name
    shutil.copy2(src, dst)

    # 提取文本内容用于检索
    text = _extract_text(dst)
    chunks = _chunk_text(text, chunk_size=500, overlap=50)

    # 保存 chunks 索引
    index_path = kb_dir / f"{dst.stem}_index.json"
    index_data = {
        "doc_name": dst_name,
        "source": str(src),
        "n_chunks": len(chunks),
        "chunks": chunks,
        "added_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    index_path.write_text(
        json.dumps(index_data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # 更新 profile
    profile = load_profile(model_path)
    docs = profile.get("knowledge_docs", [])
    docs.append({
        "name": dst_name,
        "chunks": len(chunks),
        "chars": len(text),
        "added_at": index_data["added_at"],
    })
    profile["knowledge_docs"] = docs
    save_profile(model_path, profile)

    log(f"✅ 知识文档添加: {dst_name} ({len(chunks)} chunks)")
    return {"name": dst_name, "chunks": len(chunks), "chars": len(text)}


def remove_knowledge_doc(model_path: str, doc_name: str) -> bool:
    """移除知识文档"""
    d = _model_dir(model_path)
    kb_dir = d / "knowledge_base"

    removed = False
    for f in kb_dir.glob(f"{Path(doc_name).stem}*"):
        f.unlink()
        removed = True

    if removed:
        profile = load_profile(model_path)
        profile["knowledge_docs"] = [
            doc for doc in profile.get("knowledge_docs", [])
            if doc.get("name") != doc_name
        ]
        save_profile(model_path, profile)
        log(f"✅ 知识文档移除: {doc_name}")

    return removed


def search_knowledge(model_path: str, query: str, top_k: int = 5) -> List[Dict]:
    """简单关键词检索知识库"""
    d = _model_dir(model_path)
    kb_dir = d / "knowledge_base"
    if not kb_dir.exists():
        return []

    results = []
    query_lower = query.lower()
    query_chars = set(query_lower)

    for idx_file in kb_dir.glob("*_index.json"):
        try:
            data = json.loads(idx_file.read_text(encoding="utf-8"))
            for i, chunk in enumerate(data.get("chunks", [])):
                # 简单相关性评分: 关键词重叠
                chunk_lower = chunk.lower()
                score = 0
                for word in query_lower.split():
                    if word in chunk_lower:
                        score += 1
                # 字符重叠加分
                score += len(query_chars & set(chunk_lower)) / max(len(query_chars), 1) * 0.5

                if score > 0:
                    results.append({
                        "doc": data["doc_name"],
                        "chunk_idx": i,
                        "text": chunk,
                        "score": score,
                    })
        except Exception:
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def build_rag_context(model_path: str, query: str, max_chars: int = 2000) -> str:
    """构建 RAG 上下文: 检索相关文档片段拼接"""
    hits = search_knowledge(model_path, query, top_k=8)
    if not hits:
        return ""

    context_parts = []
    total = 0
    for hit in hits:
        text = hit["text"]
        if total + len(text) > max_chars:
            break
        context_parts.append(f"[{hit['doc']}] {text}")
        total += len(text)

    return "\n---\n".join(context_parts)


# ════════════════════════════════════════════════════
#  Ollama Modelfile 生成
# ════════════════════════════════════════════════════

def generate_modelfile(model_path: str, gguf_path: str = "") -> str:
    """生成 Ollama Modelfile

    包含: 模型路径 + 系统提示词 + 推理参数 + 停止符
    """
    profile = load_profile(model_path)
    d = _model_dir(model_path)

    # 找 GGUF 文件
    if not gguf_path:
        for f in d.iterdir():
            if f.suffix == ".gguf":
                gguf_path = str(f)
                break

    lines = []

    # FROM
    if gguf_path:
        lines.append(f"FROM {gguf_path}")
    else:
        lines.append(f"# FROM <path-to-your-gguf>")
        lines.append(f"# 提示: 先用「部署导出」Tab 将模型转为 GGUF 格式")

    # 模板
    template = profile.get("chat_template", "")
    if template:
        lines.append(f'\nTEMPLATE """')
        lines.append(template)
        lines.append('"""')

    # 系统提示词
    sp = profile.get("system_prompt", "")
    if sp:
        lines.append(f'\nSYSTEM """')
        lines.append(sp)
        lines.append('"""')

    # 推理参数
    params = profile.get("parameters", {})
    lines.append("")
    if params.get("temperature", 0.7) != 0.7:
        lines.append(f"PARAMETER temperature {params['temperature']}")
    if params.get("top_p", 0.9) != 0.9:
        lines.append(f"PARAMETER top_p {params['top_p']}")
    if params.get("top_k", 50) != 50:
        lines.append(f"PARAMETER top_k {params['top_k']}")
    if params.get("repeat_penalty", 1.1) != 1.1:
        lines.append(f"PARAMETER repeat_penalty {params['repeat_penalty']}")
    if params.get("max_tokens", 2048) != 2048:
        lines.append(f"PARAMETER num_predict {params['max_tokens']}")

    # 停止符
    for stop in params.get("stop_sequences", []):
        if stop:
            lines.append(f'PARAMETER stop "{stop}"')

    # 知识库提示
    kb_docs = profile.get("knowledge_docs", [])
    if kb_docs:
        lines.append("")
        lines.append(f"# 知识库: {len(kb_docs)} 个文档")
        lines.append("# 知识库内容已嵌入系统提示词中 (RAG 模式)")

    modelfile_content = "\n".join(lines)

    # 保存
    mf_path = d / "Modelfile"
    mf_path.write_text(modelfile_content, encoding="utf-8")
    log(f"✅ Modelfile 已生成: {mf_path}")

    return modelfile_content


def generate_system_prompt_with_knowledge(model_path: str) -> str:
    """生成包含知识库摘要的系统提示词"""
    profile = load_profile(model_path)
    sp = profile.get("system_prompt", "")
    d = _model_dir(model_path)
    kb_dir = d / "knowledge_base"

    if not kb_dir.exists():
        return sp

    # 收集所有知识文档内容
    kb_summary_parts = []
    total_chars = 0
    max_kb_chars = 4000  # 限制知识库总长度

    for idx_file in sorted(kb_dir.glob("*_index.json")):
        try:
            data = json.loads(idx_file.read_text(encoding="utf-8"))
            doc_name = data.get("doc_name", idx_file.stem)
            chunks = data.get("chunks", [])
            for chunk in chunks:
                if total_chars + len(chunk) > max_kb_chars:
                    break
                kb_summary_parts.append(chunk)
                total_chars += len(chunk)
        except Exception:
            continue

    if not kb_summary_parts:
        return sp

    kb_block = "\n".join(kb_summary_parts)
    combined = f"""{sp}

<knowledge_base>
以下是你的参考知识库内容，回答问题时请优先参考这些信息:

{kb_block}
</knowledge_base>""".strip()

    return combined


# ════════════════════════════════════════════════════
#  Chat Template 编辑
# ════════════════════════════════════════════════════

def load_chat_template(model_path: str) -> str:
    """加载模型的 chat template"""
    d = _model_dir(model_path)

    # 优先从 profile 读
    profile = load_profile(model_path)
    if profile.get("chat_template"):
        return profile["chat_template"]

    # 从 tokenizer_config.json 读
    tc_path = d / "tokenizer_config.json"
    if tc_path.exists():
        try:
            tc = json.loads(tc_path.read_text(encoding="utf-8"))
            return tc.get("chat_template", "")
        except Exception:
            pass

    return ""


def save_chat_template(model_path: str, template: str) -> str:
    """保存 chat template"""
    d = _model_dir(model_path)

    # 保存到 profile
    profile = load_profile(model_path)
    profile["chat_template"] = template
    save_profile(model_path, profile)

    # 也写入 tokenizer_config.json
    tc_path = d / "tokenizer_config.json"
    if tc_path.exists():
        try:
            tc = json.loads(tc_path.read_text(encoding="utf-8"))
            tc["chat_template"] = template
            tc_path.write_text(
                json.dumps(tc, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass

    log(f"✅ Chat template 已更新: {d.name}")
    return "OK"


# ════════════════════════════════════════════════════
#  模型架构摘要
# ════════════════════════════════════════════════════

def get_model_summary(model_path: str) -> str:
    """生成模型架构摘要（丰富版）"""
    profile = load_profile(model_path)
    arch = profile.get("_architecture", {})
    params = profile.get("_estimated_params", "未知")
    kb_docs = profile.get("knowledge_docs", [])
    d = _model_dir(model_path)

    lines = [f"### 📋 模型: {profile.get('name', d.name)}"]

    if profile.get("author"):
        lines.append(f"**作者**: {profile['author']}")
    if profile.get("description"):
        lines.append(f"_{profile['description']}_")
    if profile.get("tags"):
        lines.append(f"🏷️ {', '.join(profile['tags'])}")

    lines.append("")

    # 架构信息
    if arch:
        mtype = arch.get("model_type", "?")
        layers = arch.get("num_hidden_layers", "?")
        hidden = arch.get("hidden_size", "?")
        heads = arch.get("num_attention_heads", "?")
        inter = arch.get("intermediate_size", "?")
        vocab = arch.get("vocab_size", "?")
        maxpos = arch.get("max_position_embeddings", "?")
        lines.append(f"| 架构 | {mtype} | 参数量 | ~{params} |")
        lines.append(f"|:--|:--|:--|:--|")
        lines.append(f"| 层数 | {layers} | Hidden | {hidden} |")
        lines.append(f"| 注意力头 | {heads} | MLP | {inter} |")
        lines.append(f"| 词表 | {vocab} | 最大位置 | {maxpos} |")
        # VRAM 估算
        try:
            p = float(str(params).replace("B", ""))
            vram_fp16 = p * 2
            vram_q4 = p * 0.6
            lines.append(f"\n💾 **显存估算**: FP16 ≈ {vram_fp16:.1f}GB | Q4 ≈ {vram_q4:.1f}GB")
        except Exception:
            pass
    else:
        lines.append("⚠️ _未找到 config.json，无法读取架构信息_")

    # 文件统计
    try:
        model_files = [f for f in d.iterdir() if f.suffix in (".safetensors", ".bin", ".gguf")]
        total_mb = sum(f.stat().st_size for f in model_files) / 1e6
        lines.append(f"\n📁 **磁盘**: {len(model_files)} 个权重文件 | {total_mb:,.0f} MB")
    except Exception:
        pass

    lines.append("")

    # LoRA 适配器检测
    has_lora = (d / "adapter_config.json").exists() or (d / "adapter_model.safetensors").exists()
    if has_lora:
        lines.append("🔌 **LoRA**: ✅ 检测到适配器（可合并或直接加载）")
    else:
        lines.append("🔌 **LoRA**: — 完整模型（非适配器）")

    # 编辑状态
    sp = profile.get("system_prompt", "")
    safety = profile.get("safety", {})
    template = profile.get("chat_template", "")
    status_items = []
    status_items.append(f"{'✅' if sp else '⬜'} 提示词{'(' + str(len(sp)) + '字)' if sp else ''}")
    status_items.append(f"{'✅' if kb_docs else '⬜'} 知识库{'(' + str(len(kb_docs)) + '文档)' if kb_docs else ''}")
    status_items.append(f"{'✅' if safety.get('enabled') else '⬜'} 安全护栏")
    status_items.append(f"{'✅' if template else '⬜'} 自定义模板")
    lines.append("\n**编辑状态**: " + " | ".join(status_items))

    return "\n".join(lines)


def get_model_arch_brief(model_path: str) -> str:
    """生成简洁的架构单行摘要（用于手术区等紧凑场景）"""
    try:
        d = _model_dir(model_path)
        cfg_path = d / "config.json"
        if not cfg_path.exists():
            return f"⚠️ **{d.name}** — 未找到 config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        mtype = cfg.get("model_type", "?")
        layers = cfg.get("num_hidden_layers", "?")
        hidden = cfg.get("hidden_size", "?")
        heads = cfg.get("num_attention_heads", "?")
        inter = cfg.get("intermediate_size", "?")
        vocab = cfg.get("vocab_size", "?")
        total_size = sum(
            f.stat().st_size for f in d.iterdir()
            if f.suffix in (".safetensors", ".bin")
        )
        params_str = f"{total_size / 2 / 1e9:.2f}B" if total_size > 0 else "?"
        return (f"📐 **{d.name}** | {mtype} | {layers}层 × {hidden}d | "
                f"{heads}头 | MLP {inter} | 词表 {vocab} | ~{params_str}")
    except Exception as e:
        return f"⚠️ 读取失败: {e}"


# ════════════════════════════════════════════════════
#  工具
# ════════════════════════════════════════════════════

def _extract_text(path: Path) -> str:
    """从文件提取纯文本"""
    suffix = path.suffix.lower()
    try:
        if suffix in (".txt", ".md", ".csv", ".log"):
            return path.read_text(encoding="utf-8")
        elif suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return "\n".join(str(item) for item in data)
            return json.dumps(data, ensure_ascii=False, indent=2, default=str)
        elif suffix == ".jsonl":
            lines = path.read_text(encoding="utf-8").strip().split("\n")
            return "\n".join(lines)
        elif suffix == ".pdf":
            try:
                import fitz
                doc = fitz.open(str(path))
                return "\n".join(page.get_text() for page in doc)
            except ImportError:
                return path.read_text(encoding="utf-8", errors="ignore")
        else:
            return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        log(f"提取文本失败 {path}: {e}")
        return ""


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """文本分块"""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


# ════════════════════════════════════════════════════
#  角色预设 (Persona Presets)
# ════════════════════════════════════════════════════

def save_preset(model_path: str, preset_name: str) -> str:
    """将当前配置另存为预设"""
    profile = load_profile(model_path)
    preset = {
        "name": preset_name,
        "system_prompt": profile.get("system_prompt", ""),
        "parameters": profile.get("parameters", {}),
        "safety": profile.get("safety", {}),
        "output_format": profile.get("output_format", {}),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    presets = profile.get("persona_presets", [])
    # 替换同名
    presets = [p for p in presets if p.get("name") != preset_name]
    presets.append(preset)
    profile["persona_presets"] = presets
    save_profile(model_path, profile)
    return f"✅ 预设已保存: {preset_name}"


def load_preset(model_path: str, preset_name: str) -> Dict:
    """加载预设到当前配置"""
    profile = load_profile(model_path)
    for preset in profile.get("persona_presets", []):
        if preset.get("name") == preset_name:
            profile["system_prompt"] = preset.get("system_prompt", profile["system_prompt"])
            profile["parameters"] = preset.get("parameters", profile["parameters"])
            profile["safety"] = preset.get("safety", profile.get("safety", {}))
            profile["output_format"] = preset.get("output_format", profile.get("output_format", {}))
            profile["active_preset"] = preset_name
            save_profile(model_path, profile)
            return profile
    raise ValueError(f"预设不存在: {preset_name}")


def delete_preset(model_path: str, preset_name: str) -> str:
    profile = load_profile(model_path)
    profile["persona_presets"] = [
        p for p in profile.get("persona_presets", []) if p.get("name") != preset_name
    ]
    save_profile(model_path, profile)
    return f"✅ 已删除预设: {preset_name}"


def list_presets(model_path: str) -> List[str]:
    profile = load_profile(model_path)
    return [p["name"] for p in profile.get("persona_presets", []) if p.get("name")]


# ════════════════════════════════════════════════════
#  烘焙: 将所有编辑写入模型文件
# ════════════════════════════════════════════════════

def bake_profile_into_model(model_path: str, task=None) -> str:
    """将所有编辑烘焙（bake）进模型文件

    把 profile 中的设定写入实际的模型配置文件:
      1. system_prompt + knowledge → tokenizer_config.json (chat_template)
      2. parameters → generation_config.json
      3. safety + output_format → 融入 system_prompt
      4. 写 system_prompt.txt 兼容文件
      5. 生成 Modelfile (Ollama)

    烘焙后模型是"开箱即用"的 — 任何工具加载都能自动带上这些设定。
    """
    from core import log
    _safe_update = lambda t, p, m: t.update_progress(p, m) if t else log(m)

    d = _model_dir(model_path)
    profile = load_profile(model_path)
    _safe_update(task, 5, "读取模型档案...")

    # ── 1. 构建完整系统提示词 ──
    _safe_update(task, 10, "构建系统提示词...")
    full_sp = _build_full_system_prompt(profile, d)

    # ── 2. 写入 tokenizer_config.json ──
    _safe_update(task, 25, "写入 tokenizer_config.json...")
    tc_path = d / "tokenizer_config.json"
    if tc_path.exists():
        try:
            tc = json.loads(tc_path.read_text(encoding="utf-8"))
        except Exception:
            tc = {}
    else:
        tc = {}

    # 如果有自定义 chat_template，使用它；否则注入 system prompt 到默认模板
    custom_tpl = profile.get("chat_template", "")
    if custom_tpl:
        tc["chat_template"] = custom_tpl
    elif full_sp:
        # 为没有 chat_template 的模型，注入一个带系统提示的基础模板
        existing_tpl = tc.get("chat_template", "")
        if not existing_tpl:
            tc["chat_template"] = _make_default_chat_template(full_sp)

    # 写入 eos/bos 相关
    params = profile.get("parameters", {})
    stops = params.get("stop_sequences", [])
    if stops:
        tc["eos_token"] = stops[0] if len(stops) == 1 else tc.get("eos_token", "</s>")

    tc_path.write_text(json.dumps(tc, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"  ✅ tokenizer_config.json 已更新")

    # ── 3. 写入 generation_config.json ──
    _safe_update(task, 40, "写入 generation_config.json...")
    gc_path = d / "generation_config.json"
    try:
        gc = json.loads(gc_path.read_text(encoding="utf-8")) if gc_path.exists() else {}
    except Exception:
        gc = {}

    gc["temperature"] = params.get("temperature", 0.7)
    gc["top_p"] = params.get("top_p", 0.9)
    gc["top_k"] = params.get("top_k", 50)
    gc["repetition_penalty"] = params.get("repeat_penalty", 1.1)
    gc["max_new_tokens"] = params.get("max_tokens", 2048)
    if stops:
        # 尝试转为 token IDs (如果 tokenizer 存在)
        gc["stop_strings"] = stops

    gc_path.write_text(json.dumps(gc, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"  ✅ generation_config.json 已更新")

    # ── 4. 写 system_prompt.txt ──
    _safe_update(task, 55, "写入 system_prompt.txt...")
    (d / "system_prompt.txt").write_text(full_sp, encoding="utf-8")

    # ── 5. 生成 Modelfile ──
    _safe_update(task, 65, "生成 Modelfile...")
    generate_modelfile(model_path)

    # ── 6. 写 README.md (模型卡) ──
    _safe_update(task, 75, "生成 README.md...")
    _write_model_card(d, profile, full_sp)

    # ── 7. 保存烘焙元信息 ──
    bake_meta = {
        "baked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_prompt_length": len(full_sp),
        "knowledge_docs": len(profile.get("knowledge_docs", [])),
        "safety_enabled": profile.get("safety", {}).get("enabled", False),
        "output_format": profile.get("output_format", {}).get("mode", "free"),
    }
    (d / "forgex_bake_meta.json").write_text(
        json.dumps(bake_meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    _safe_update(task, 100, f"✅ 烘焙完成: {d.name} | SP={len(full_sp)}字")
    return str(d)


def _build_full_system_prompt(profile: Dict, model_dir: Path) -> str:
    """构建完整系统提示词 = 基础SP + 安全护栏 + 输出格式 + 知识库"""
    parts = []

    # 基础系统提示词
    base_sp = profile.get("system_prompt", "")
    if base_sp:
        parts.append(base_sp)

    # 安全护栏
    safety = profile.get("safety", {})
    if safety.get("enabled"):
        safety_rules = []
        refusal = safety.get("refusal_topics", [])
        if refusal:
            topics_str = "、".join(refusal)
            safety_rules.append(f"你绝对不能讨论或回答以下主题相关的问题: {topics_str}。如果用户询问这些话题，礼貌地拒绝并解释你无法帮助。")
        cf = safety.get("content_filter", "")
        if cf:
            safety_rules.append(cf)
        if safety.get("require_disclaimer") and safety.get("disclaimer_text"):
            safety_rules.append(f"在每次回答结束时，附加以下免责声明:\n{safety['disclaimer_text']}")

        if safety_rules:
            parts.append("\n<safety_rules>\n" + "\n".join(safety_rules) + "\n</safety_rules>")

    # 输出格式
    fmt = profile.get("output_format", {})
    mode = fmt.get("mode", "free")
    if mode != "free":
        fmt_rules = []
        if mode == "json":
            schema = fmt.get("json_schema", "")
            fmt_rules.append("你必须以 JSON 格式回答所有问题，不要输出任何 JSON 之外的文字。")
            if schema:
                fmt_rules.append(f"JSON 必须符合以下 Schema:\n```json\n{schema}\n```")
        elif mode == "markdown":
            fmt_rules.append("所有回答使用 Markdown 格式，包含适当的标题、列表和代码块。")
        elif mode == "custom":
            tpl = fmt.get("custom_template", "")
            if tpl:
                fmt_rules.append(f"回答必须严格按照以下模板格式:\n{tpl}")
        lang = fmt.get("language", "")
        if lang:
            fmt_rules.append(f"你必须始终使用 {lang} 回答。")
        max_p = fmt.get("max_paragraphs", 0)
        if max_p > 0:
            fmt_rules.append(f"回答不超过 {max_p} 段。")
        if fmt_rules:
            parts.append("\n<output_format>\n" + "\n".join(fmt_rules) + "\n</output_format>")

    # 知识库
    kb_dir = model_dir / "knowledge_base"
    if kb_dir.exists():
        kb_parts = []
        total = 0
        max_kb = 4000
        for idx_file in sorted(kb_dir.glob("*_index.json")):
            try:
                data = json.loads(idx_file.read_text(encoding="utf-8"))
                for chunk in data.get("chunks", []):
                    if total + len(chunk) > max_kb:
                        break
                    kb_parts.append(chunk)
                    total += len(chunk)
            except Exception:
                continue
        if kb_parts:
            kb_text = "\n".join(kb_parts)
            parts.append(f"\n<knowledge_base>\n以下是你的参考知识库，回答时请优先参考:\n{kb_text}\n</knowledge_base>")

    return "\n\n".join(parts).strip()


def _make_default_chat_template(system_prompt: str) -> str:
    """生成包含系统提示词的默认 chat template"""
    # 使用 ChatML 格式，系统提示词硬编码
    sp_escaped = system_prompt.replace("'", "\\'").replace('"', '\\"')
    return (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'user' %}"
        "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'assistant' %}"
        "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "<|im_start|>assistant\n"
        "{% endif %}"
    )


def _write_model_card(model_dir: Path, profile: Dict, full_sp: str):
    """生成 README.md 模型卡"""
    arch = profile.get("_architecture", {})
    params_est = profile.get("_estimated_params", "未知")
    kb_docs = profile.get("knowledge_docs", [])
    params = profile.get("parameters", {})

    lines = [
        f"# {profile.get('name', model_dir.name)}",
        "",
        f"> {profile.get('description', '由 ForgeX 训练和编辑的模型')}",
        "",
        "## 基本信息",
        f"- **作者**: {profile.get('author', 'ForgeX')}",
        f"- **标签**: {', '.join(profile.get('tags', []))}",
        f"- **用途**: {profile.get('use_case', '')}",
        f"- **参数量**: ~{params_est}",
        "",
    ]

    if arch:
        lines.extend([
            "## 架构",
            f"- 类型: {arch.get('model_type', '?')}",
            f"- 层数: {arch.get('num_hidden_layers', '?')}",
            f"- Hidden: {arch.get('hidden_size', '?')}",
            f"- Heads: {arch.get('num_attention_heads', '?')}",
            f"- 词表: {arch.get('vocab_size', '?')}",
            "",
        ])

    lines.extend([
        "## 推理参数",
        f"- Temperature: {params.get('temperature', 0.7)}",
        f"- Top-P: {params.get('top_p', 0.9)}",
        f"- Top-K: {params.get('top_k', 50)}",
        f"- 重复惩罚: {params.get('repeat_penalty', 1.1)}",
        f"- 最大 Token: {params.get('max_tokens', 2048)}",
        "",
    ])

    if kb_docs:
        lines.extend([
            "## 知识库",
            f"模型包含 {len(kb_docs)} 个参考文档:",
        ])
        for doc in kb_docs:
            lines.append(f"- {doc.get('name', '?')} ({doc.get('chunks', 0)} chunks)")
        lines.append("")

    safety = profile.get("safety", {})
    if safety.get("enabled"):
        lines.extend([
            "## 安全设置",
            f"- 拒绝主题: {', '.join(safety.get('refusal_topics', []))}",
            f"- 免责声明: {'启用' if safety.get('require_disclaimer') else '未启用'}",
            "",
        ])

    lines.extend([
        "---",
        f"*由 ForgeX 生成于 {time.strftime('%Y-%m-%d %H:%M')}*",
    ])

    (model_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


# ════════════════════════════════════════════════════
#  一键烘焙导出: Bake → Merge → GGUF → Modelfile
# ════════════════════════════════════════════════════

def bake_and_export(
    model_path: str,
    output_name: str = "",
    quant_type: str = "Q4_K_M",
    ollama_import: bool = False,
    task=None,
) -> Dict[str, str]:
    """一键烘焙导出: 编辑 → 烘焙 → 合并LoRA → GGUF → Modelfile → Ollama

    全自动流水线，把所有编辑打包成可部署的 GGUF 模型。

    Returns: {"model_dir": ..., "gguf_path": ..., "modelfile": ..., "ollama": ...}
    """
    from core import log
    _upd = lambda p, m: task.update_progress(p, m) if task else log(m)

    results = {}

    # Step 1: 烘焙
    _upd(5, "Step 1/4: 烘焙编辑到模型文件...")
    baked_dir = bake_profile_into_model(model_path)
    results["model_dir"] = baked_dir
    d = Path(baked_dir)

    # Step 2: 合并 LoRA (如果是 LoRA 适配器)
    is_lora = (d / "adapter_config.json").exists() and not (d / "config.json").exists()
    if is_lora:
        _upd(20, "Step 2/4: 检测到 LoRA，自动合并...")
        try:
            adapter_cfg = json.loads((d / "adapter_config.json").read_text(encoding="utf-8"))
            base_model = adapter_cfg.get("base_model_name_or_path", "")
            if base_model:
                from core.merger import merger
                merge_name = (output_name or d.name) + "_merged"
                merged = merger.merge_lora_to_base(
                    base_model=base_model,
                    lora_path=str(d),
                    output_name=merge_name,
                    task=task,
                )
                # 把烘焙文件复制到合并后目录
                merged_dir = Path(merged)
                for f in ["system_prompt.txt", META_FILE, "forgex_bake_meta.json",
                          "Modelfile", "README.md"]:
                    src = d / f
                    if src.exists():
                        shutil.copy2(src, merged_dir / f)
                # 复制知识库
                kb_src = d / "knowledge_base"
                if kb_src.exists():
                    kb_dst = merged_dir / "knowledge_base"
                    if kb_dst.exists():
                        shutil.rmtree(kb_dst)
                    shutil.copytree(kb_src, kb_dst)
                # 复制 generation_config
                gc_src = d / "generation_config.json"
                if gc_src.exists():
                    shutil.copy2(gc_src, merged_dir / "generation_config.json")

                d = merged_dir
                results["model_dir"] = str(d)
                log(f"  ✅ LoRA 合并完成: {d}")
            else:
                _upd(25, "⚠️ 未找到基座模型路径，跳过合并")
        except Exception as e:
            _upd(25, f"⚠️ LoRA 合并失败: {e}，继续尝试导出")
    else:
        _upd(25, "Step 2/4: 完整模型，无需合并")

    # Step 3: GGUF 导出
    _upd(35, f"Step 3/4: 导出 GGUF ({quant_type})...")
    try:
        from core.exporter import Exporter
        exp = Exporter()
        gguf_path = exp.export_gguf(
            model_path=str(d),
            quant_type=quant_type,
            output_name=output_name or d.name,
            task=task,
        )
        results["gguf_path"] = str(gguf_path)
        log(f"  ✅ GGUF 导出: {gguf_path}")
    except Exception as e:
        results["gguf_error"] = str(e)
        _upd(70, f"⚠️ GGUF 导出失败: {e}")

    # Step 4: Modelfile + Ollama
    _upd(85, "Step 4/4: 生成 Modelfile...")
    gguf_file = results.get("gguf_path", "")
    modelfile_content = generate_modelfile(model_path, gguf_file)
    results["modelfile"] = modelfile_content

    if ollama_import and gguf_file:
        _upd(90, "导入 Ollama...")
        try:
            import subprocess
            ollama_name = output_name or d.name
            mf_path = d / "Modelfile"
            cmd = ["ollama", "create", ollama_name, "-f", str(mf_path)]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if proc.returncode == 0:
                results["ollama"] = f"✅ 已导入 Ollama: {ollama_name}"
            else:
                results["ollama"] = f"⚠️ Ollama 导入失败: {proc.stderr}"
        except Exception as e:
            results["ollama"] = f"⚠️ {e}"

    _upd(100, f"✅ 烘焙导出完成: {d.name}")
    return results
