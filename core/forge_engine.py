# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

"""ForgeX 锻造引擎 — 让模型自我进化的高级训练流水线。

核心能力:
  1. 合成数据工厂: 用运行中的模型生成高质量训练数据
  2. 自进化循环:   生成→训练→生成→训练… (Self-Play / Iterative Refinement)
  3. 一键锻造:     文档→提取要点→生成 QA→训练 LoRA
  4. 聊天提取:     从对话历史自动提取 instruction/output 对
  5. 锦标赛进化:   多个 LoRA 竞争评测，优胜者杂交出更强后代
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core import DATASETS_DIR, LORAS_DIR, log


class ForgeEngine:
    """锻造引擎 — 高级训练流水线"""

    def __init__(self):
        self._stop = False
        self._running = False

    def stop(self):
        self._stop = True

    @property
    def is_running(self):
        return self._running

    # ================================================================
    #  1. 合成数据工厂
    # ================================================================
    def generate_synthetic_data(
        self,
        chat_fn: Callable[[str], str],
        topic: str = "通用",
        count: int = 100,
        style: str = "instruction",
        progress_cb: Callable = None,
        task=None,
    ) -> Optional[Path]:
        """用推理引擎生成合成训练数据。

        Args:
            chat_fn:  接收 prompt 字符串，返回模型回复字符串
            topic:    数据主题/风格描述
            count:    生成条数
            style:    "instruction" (指令对) | "conversation" (多轮对话) | "text" (纯文本)
            progress_cb: 进度回调 (value, status_text)
        Returns:
            生成的 JSONL 文件路径
        """
        self._running = True
        self._stop = False
        prog = progress_cb or (lambda v, s: None)

        ts = time.strftime("%Y%m%d_%H%M%S")
        out = Path(DATASETS_DIR) / f"synthetic_{topic[:20]}_{ts}.jsonl"

        prompts_instruction = [
            f"关于「{topic}」，请生成一个高质量的问答对。格式:\n问题: ...\n回答: ...",
            f"请针对「{topic}」领域，出一道需要深入思考的题目，并给出详细解答。",
            f"请模拟一个关于「{topic}」的真实用户提问，并给出专业回答。",
            f"生成一段关于「{topic}」的教学内容，包含概念解释和具体例子。",
            f"关于「{topic}」，提出一个常见误区并加以纠正。",
            f"请写一个关于「{topic}」的实用技巧或最佳实践。",
        ]
        prompts_conversation = [
            f"请模拟一段关于「{topic}」的自然对话，包含 2-3 轮问答。",
            f"写一段关于「{topic}」的师生对话，学生提问老师解答。",
        ]
        prompts_text = [
            f"请写一段关于「{topic}」的高质量文章段落（200-500字）。",
            f"请详细介绍「{topic}」的核心概念。",
        ]

        prompt_pool = {
            "instruction": prompts_instruction,
            "conversation": prompts_conversation,
            "text": prompts_text,
        }.get(style, prompts_instruction)

        data = []
        for i in range(count):
            if self._stop:
                break
            prog(i / count, f"合成数据 {i+1}/{count}")
            if task:
                _safe_update(task, 30 + 60 * i / count, f"合成 {i+1}/{count}")

            prompt = random.choice(prompt_pool)
            try:
                reply = chat_fn(prompt)
                if not reply or len(reply.strip()) < 10:
                    continue

                if style == "instruction":
                    # 尝试解析 问题/回答 格式
                    entry = _parse_qa(reply, prompt)
                elif style == "conversation":
                    entry = {"messages": _parse_conversation(reply)}
                else:
                    entry = {"text": reply.strip()}
                data.append(entry)
            except Exception as e:
                log(f"合成第 {i+1} 条失败: {e}")

        if not data:
            log("⚠️ 合成数据为空")
            self._running = False
            return None

        with open(out, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

        log(f"✅ 合成数据: {out.name} ({len(data)} 条)")
        self._stop = False
        self._running = False
        return out

    # ================================================================
    #  2. 自进化循环
    # ================================================================
    def self_evolution(
        self,
        trainer_engine,
        base_model: str,
        chat_fn: Callable[[str], str],
        topic: str = "通用",
        generations: int = 3,
        synth_count: int = 50,
        train_params: Dict = None,
        progress_cb: Callable = None,
        task=None,
    ) -> Optional[str]:
        """自进化循环: 生成合成数据→训练→用新模型再生成→再训练…

        每一代:
          1. 用当前模型生成合成数据
          2. 与之前代的数据合并
          3. 训练新 LoRA
          4. 加载新 LoRA，生成下一代数据

        Args:
            trainer_engine:  TrainerEngine 实例
            base_model:      基座模型 ID
            chat_fn:         推理函数 (首代使用)
            topic:           数据主题
            generations:     进化代数
            synth_count:     每代合成条数
            train_params:    训练参数 dict
        Returns:
            最终 LoRA 输出目录
        """
        self._running = True
        self._stop = False
        prog = progress_cb or (lambda v, s: None)
        params = dict(train_params or {})
        all_data_files = []
        result_path = None

        for g in range(1, generations + 1):
            if self._stop:
                log("自进化已中止")
                break

            prog(g / (generations + 1), f"🔄 自进化 G{g}/{generations}")
            log(f"══ 自进化 G{g}/{generations} ══")

            # 1. 生成合成数据
            synth_path = self.generate_synthetic_data(
                chat_fn=chat_fn, topic=f"{topic}_G{g}",
                count=synth_count, style="instruction",
                progress_cb=progress_cb, task=task,
            )
            if synth_path:
                all_data_files.append(str(synth_path))

            if not all_data_files:
                log(f"G{g}: 无数据可用，跳过训练")
                continue

            # 2. 训练
            params["output_name"] = params.get("output_name", "evo") + f"_G{g}"
            params["epochs"] = max(1, params.get("epochs", 1))
            try:
                result_path = trainer_engine.train(
                    method="sft",
                    base_model=base_model,
                    dataset_path=all_data_files,
                    params=params,
                    task=task,
                )
                log(f"G{g}: LoRA 训练完成 → {result_path}")
            except Exception as e:
                log(f"G{g}: 训练失败 - {e}")
                break

            # TODO: 未来可以在这里加载新 LoRA 更新 chat_fn

        self._stop = False
        self._running = False
        return result_path

    # ================================================================
    #  3. 一键锻造 (文档→合成QA→训练)
    # ================================================================
    def one_click_forge(
        self,
        trainer_engine,
        base_model: str,
        doc_paths: List[str],
        chat_fn: Callable[[str], str],
        output_name: str = "forged_lora",
        qa_count: int = 100,
        train_params: Dict = None,
        progress_cb: Callable = None,
        task=None,
    ) -> Optional[str]:
        """一键锻造: 从文档提取内容 → 用模型生成QA对 → 训练 LoRA。

        适用场景: 把个人资料/公司文档/专业教材变成模型的知识。
        """
        self._running = True
        self._stop = False
        prog = progress_cb or (lambda v, s: None)

        # 1. 提取文档文本
        prog(0.1, "提取文档内容...")
        all_text = ""
        for dp in doc_paths:
            p = Path(dp)
            if p.is_file():
                all_text += f"\n--- {p.name} ---\n" + _extract_text(p)
            elif p.is_dir():
                for f in p.rglob("*.*"):
                    if f.suffix.lower() in {".txt", ".md", ".csv", ".jsonl", ".json"}:
                        all_text += f"\n--- {f.name} ---\n" + _extract_text(f)[:10000]

        if not all_text.strip():
            log("❌ 无法从文档中提取有效内容")
            self._running = False
            return None
        log(f"提取文本: {len(all_text)} 字符")

        # 2. 生成 QA 训练数据
        prog(0.3, "生成训练数据...")
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = Path(DATASETS_DIR) / f"forge_{output_name}_{ts}.jsonl"

        # 将文档切分为段落
        chunks = _split_text(all_text, chunk_size=2000, overlap=200)
        data = []

        for idx, chunk in enumerate(chunks):
            if self._stop or len(data) >= qa_count:
                break
            prog(0.3 + 0.4 * idx / max(len(chunks), 1), f"生成 QA {len(data)}/{qa_count}")

            # 每个 chunk 生成多个 QA
            for _ in range(min(3, qa_count - len(data))):
                if self._stop:
                    break
                prompt = (
                    f"基于以下内容，生成一个高质量问答对。\n\n"
                    f"内容:\n{chunk[:1500]}\n\n"
                    f"请用以下格式回复:\n问题: ...\n回答: ..."
                )
                try:
                    reply = chat_fn(prompt)
                    entry = _parse_qa(reply, prompt)
                    if entry.get("instruction") and entry.get("output"):
                        data.append(entry)
                except Exception:
                    pass

        if not data:
            log("❌ 未能生成有效训练数据")
            self._running = False
            return None

        with open(out, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        log(f"✅ 生成 {len(data)} 条 QA 训练数据 → {out.name}")

        # 3. 训练 LoRA
        prog(0.7, "开始训练...")
        params = dict(train_params or {})
        params["output_name"] = output_name

        try:
            result = trainer_engine.train(
                method="sft", base_model=base_model,
                dataset_path=[str(out)], params=params, task=task,
            )
            self._running = False
            return result
        except Exception as e:
            log(f"❌ 训练失败: {e}")
            self._running = False
            return None

    # ================================================================
    #  4. 聊天历史 → 训练数据
    # ================================================================
    @staticmethod
    def extract_training_from_chat(
        chat_history: List[Dict],
        output_name: str = "chat_extracted",
    ) -> Optional[Path]:
        """从聊天历史提取 instruction/output 对。

        Args:
            chat_history: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        Returns:
            JSONL 文件路径
        """
        pairs = []
        history = list(chat_history)
        for i, msg in enumerate(history):
            if msg.get("role") == "user":
                for j in range(i + 1, len(history)):
                    if history[j].get("role") == "assistant":
                        user_content = msg.get("content", "").strip()
                        asst_content = history[j].get("content", "").strip()
                        if user_content and asst_content and len(asst_content) > 10:
                            pairs.append({
                                "instruction": user_content,
                                "output": asst_content,
                            })
                        break

        if not pairs:
            log("⚠️ 聊天历史中无可用对话对")
            return None

        out = Path(DATASETS_DIR) / f"{output_name}.jsonl"
        with open(out, "a", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False, default=str) + "\n")
        log(f"✅ 从聊天提取 {len(pairs)} 条训练数据 → {out.name}")
        return out

    # ================================================================
    #  5. 锦标赛进化 (多 LoRA 竞争)
    # ================================================================
    def tournament_evolution(
        self,
        lora_paths: List[str],
        eval_fn: Callable[[str], float],
        merge_fn: Callable[[str, str, float], str],
        generations: int = 3,
        progress_cb: Callable = None,
    ) -> Optional[str]:
        """锦标赛进化: 多个 LoRA 评测→淘汰→合并→变异→评测…

        Args:
            lora_paths:  LoRA 目录列表
            eval_fn:     评测函数 (lora_path → score)
            merge_fn:    合并函数 (lora1, lora2, ratio → new_lora_path)
            generations: 进化代数
        Returns:
            最优 LoRA 路径
        """
        self._running = True
        self._stop = False
        prog = progress_cb or (lambda v, s: None)

        pool = [(p, 0.0) for p in lora_paths]

        for g in range(1, generations + 1):
            if self._stop or len(pool) < 2:
                break
            prog(g / generations, f"🏟️ 锦标赛 G{g}/{generations}")

            # 评测
            scored = []
            for path, _ in pool:
                try:
                    score = eval_fn(path)
                    scored.append((path, score))
                except Exception:
                    scored.append((path, 0.0))

            scored.sort(key=lambda x: x[1], reverse=True)
            log(f"G{g}: 排名 " + ", ".join(f"{Path(p).name}={s:.2f}" for p, s in scored[:5]))

            # 保留前半
            survivors = scored[:max(2, len(scored) // 2)]

            # 杂交
            new_pool = list(survivors)
            for i in range(0, len(survivors) - 1, 2):
                try:
                    ratio = random.uniform(0.3, 0.7)
                    child_path = merge_fn(survivors[i][0], survivors[i+1][0], ratio)
                    new_pool.append((child_path, 0.0))
                except Exception as e:
                    log(f"合并失败: {e}")

            pool = new_pool

        self._stop = False
        self._running = False

        if pool:
            best = max(pool, key=lambda x: x[1])
            log(f"🏆 锦标赛冠军: {Path(best[0]).name} (score={best[1]:.2f})")
            return best[0]
        return None


# ── 工具函数 ──

def _safe_update(task, p, msg):
    if task is not None:
        try:
            task.update_progress(float(p), str(msg))
        except Exception:
            pass


def _extract_text(path: Path, max_len: int = 20000) -> str:
    """从文件提取文本"""
    try:
        suffix = path.suffix.lower()
        if suffix in (".txt", ".md", ".csv", ".jsonl", ".json"):
            return path.read_text(encoding="utf-8-sig", errors="replace")[:max_len]
        elif suffix == ".pdf":
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(str(path))
                text = "\n".join(page.get_text() for page in doc)
                doc.close()
                return text[:max_len]
            except ImportError:
                return ""
        elif suffix == ".docx":
            try:
                from docx import Document
                doc = Document(str(path))
                return "\n".join(p.text for p in doc.paragraphs)[:max_len]
            except ImportError:
                return ""
        return ""
    except Exception:
        return ""


def _split_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    """将文本分割为重叠的段落"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


def _parse_qa(text: str, fallback_instruction: str = "") -> Dict:
    """解析 '问题: ...\n回答: ...' 格式"""
    instruction = ""
    output = ""

    for line in text.split("\n"):
        line = line.strip()
        low = line.lower()
        if low.startswith(("问题:", "问题：", "q:", "question:")):
            instruction = line.split(":", 1)[-1].strip()
            if not instruction:
                instruction = line.split("：", 1)[-1].strip()
        elif low.startswith(("回答:", "回答：", "a:", "answer:")):
            output = line.split(":", 1)[-1].strip()
            if not output:
                output = line.split("：", 1)[-1].strip()

    if not instruction and not output:
        # 无法解析，整段作为 output
        return {"instruction": fallback_instruction[:200], "output": text.strip()}

    return {"instruction": instruction or fallback_instruction[:200],
            "output": output or text.strip()}


def _parse_conversation(text: str) -> List[Dict]:
    """解析多轮对话文本为 messages 格式"""
    messages = []
    current_role = None
    current_content = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        low = line.lower()

        new_role = None
        if low.startswith(("用户:", "用户：", "user:", "human:")):
            new_role = "user"
            line = line.split(":", 1)[-1].strip() or line.split("：", 1)[-1].strip()
        elif low.startswith(("助手:", "助手：", "assistant:", "ai:")):
            new_role = "assistant"
            line = line.split(":", 1)[-1].strip() or line.split("：", 1)[-1].strip()

        if new_role:
            if current_role and current_content:
                messages.append({"role": current_role, "content": "\n".join(current_content)})
            current_role = new_role
            current_content = [line] if line else []
        elif current_role:
            current_content.append(line)

    if current_role and current_content:
        messages.append({"role": current_role, "content": "\n".join(current_content)})

    return messages or [{"role": "user", "content": text[:200]},
                        {"role": "assistant", "content": text[200:]}]


forge_engine = ForgeEngine()
