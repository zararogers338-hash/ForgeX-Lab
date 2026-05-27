# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

"""ForgeX RLHF 反馈管理器 — 从聊天收集偏好数据，构建 DPO 训练对。

工作流:
1. 用户在"聊天测试"中点 👍/👎 → record()
2. 用户 A/B 比较两个回复 → record_preference()
3. 训练时调用 export_dpo_dataset() → 生成 DPO 训练用 JSONL
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from core import DATASETS_DIR, log


FEEDBACK_DIR = Path(DATASETS_DIR).parent / "feedback"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_FILE = FEEDBACK_DIR / "rlhf_feedback.jsonl"


class FeedbackManager:

    def __init__(self):
        self._count = 0

    def record(self, action: str, user_msg: str, assistant_msg: str,
               model_name: str = "", lora_name: str = ""):
        """记录单条反馈 (positive / negative)"""
        entry = {
            "action": action,
            "prompt": user_msg,
            "response": assistant_msg,
            "model": model_name,
            "lora": lora_name,
            "time": datetime.now().isoformat(),
        }
        _append_jsonl(FEEDBACK_FILE, entry)
        self._count += 1
        log(f"反馈已记录: {action} (model={model_name})")

    def record_preference(self, prompt: str, chosen: str, rejected: str,
                          model_name: str = ""):
        """记录偏好对 (A/B 比较)"""
        entry = {
            "action": "preference",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "model": model_name,
            "time": datetime.now().isoformat(),
        }
        _append_jsonl(FEEDBACK_FILE, entry)
        self._count += 1

    def get_stats(self) -> Dict[str, int]:
        stats = {"positive": 0, "negative": 0, "preference": 0, "total": 0}
        for entry in _read_jsonl(FEEDBACK_FILE):
            action = entry.get("action", "")
            if action in stats:
                stats[action] += 1
            stats["total"] += 1
        return stats

    def build_dpo_pairs(self, min_pairs: int = 2) -> List[Dict]:
        """从正负反馈自动构建 DPO 训练对。

        策略:
        1. 直接偏好对 (preference 记录) 直接使用
        2. 正负反馈配对: 同一 prompt 的 positive 回复 vs negative 回复
        3. 跨 prompt 配对: 如果同 prompt 不够，用正面回复 vs 负面回复交叉配对
        """
        preferences = []
        positive_by_prompt = {}
        negative_by_prompt = {}
        positives = []
        negatives = []

        for entry in _read_jsonl(FEEDBACK_FILE):
            action = entry.get("action", "")
            prompt = entry.get("prompt", "")
            response = entry.get("response", "")

            if action == "preference":
                preferences.append({
                    "prompt": entry.get("prompt", "请回复:"),
                    "chosen": entry.get("chosen", ""),
                    "rejected": entry.get("rejected", ""),
                })
            elif action == "positive" and response:
                positive_by_prompt.setdefault(prompt, []).append(response)
                positives.append({"prompt": prompt, "response": response})
            elif action == "negative" and response:
                negative_by_prompt.setdefault(prompt, []).append(response)
                negatives.append({"prompt": prompt, "response": response})

        pairs = list(preferences)

        # 同 prompt 配对
        for prompt in positive_by_prompt:
            if prompt in negative_by_prompt:
                pos_list = positive_by_prompt[prompt]
                neg_list = negative_by_prompt[prompt]
                for i in range(min(len(pos_list), len(neg_list))):
                    pairs.append({
                        "prompt": prompt,
                        "chosen": pos_list[i],
                        "rejected": neg_list[i],
                    })

        # 跨 prompt 交叉配对 (补充)
        if len(pairs) < min_pairs:
            for i in range(min(len(positives), len(negatives))):
                p = positives[i]
                n = negatives[i]
                if p["prompt"] != n["prompt"]:
                    pairs.append({
                        "prompt": p["prompt"] or n["prompt"] or "请回复:",
                        "chosen": p["response"],
                        "rejected": n["response"],
                    })

        return pairs

    def export_dpo_dataset(self, output_name: str = "rlhf_dpo") -> Optional[Path]:
        """导出 DPO 数据集到 datasets 目录"""
        pairs = self.build_dpo_pairs()
        if not pairs:
            log("⚠️ 无足够反馈数据导出 DPO 数据集")
            return None

        out = Path(DATASETS_DIR) / f"{output_name}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False, default=str) + "\n")
        log(f"✅ DPO 数据集已导出: {out.name} ({len(pairs)} 对)")
        return out

    def export_chat_as_sft(self, chat_history: List[Dict],
                           output_name: str = "chat_sft") -> Optional[Path]:
        """从聊天历史提取 instruction/response 对作为 SFT 训练数据。

        chat_history: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        """
        pairs = []
        history = list(chat_history)
        for i, msg in enumerate(history):
            if msg.get("role") == "user":
                for j in range(i + 1, len(history)):
                    if history[j].get("role") == "assistant":
                        pairs.append({
                            "instruction": msg["content"],
                            "output": history[j]["content"],
                        })
                        break

        if not pairs:
            log("⚠️ 聊天历史中无可用对话对")
            return None

        out = Path(DATASETS_DIR) / f"{output_name}.jsonl"
        # 追加模式
        with open(out, "a", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False, default=str) + "\n")
        log(f"✅ 从聊天提取 {len(pairs)} 条训练数据 → {out.name}")
        return out


# ── 工具函数 ──

def _append_jsonl(path: Path, entry: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass
    return out


feedback_manager = FeedbackManager()
