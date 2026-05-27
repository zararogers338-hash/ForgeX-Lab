# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

# core/ai_synthesizer.py - ForgeX v2 AI 數據合成引擎（LLM 驅動）
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from core import DATASETS_DIR, log, get_timestamp
from core.task_queue import Task


class AISynthesizer:
    """用 LLM 生成高質量訓練數據（支持 Ollama / OpenAI API）"""

    def generate_sft_data(
        self,
        topic: str,
        count: int = 100,
        style: str = "informative",
        difficulty: str = "medium",
        output_format: str = "alpaca",
        api_type: str = "ollama",
        api_base: str = "http://localhost:11434",
        api_key: str = "",
        model: str = "qwen2.5:7b",
        task: Optional[Task] = None,
    ) -> str:
        """用 LLM 生成 SFT 訓練數據"""
        progress_cb = task.update_progress if task else lambda p, m="": log(m)

        out_name = f"ai_synth_{topic}_{count}_{get_timestamp()}.jsonl"
        out_path = DATASETS_DIR / out_name

        system_prompt = self._build_system_prompt(topic, style, difficulty, output_format)
        generated = 0
        batch_size = min(5, count)

        with out_path.open("w", encoding="utf-8") as f:
            while generated < count:
                pct = (generated / count) * 90 + 5
                progress_cb(pct, f"生成中: {generated}/{count}")

                remaining = min(batch_size, count - generated)
                user_prompt = (
                    f"請生成 {remaining} 條關於「{topic}」的訓練數據。"
                    f"每條一個 JSON 對象，每行一個（JSONL 格式）。"
                    f"確保內容多樣、有深度、不重複。"
                    f"風格: {style} | 難度: {difficulty}"
                )

                try:
                    response = self._call_llm(api_type, api_base, api_key, model, system_prompt, user_prompt)
                    samples = self._parse_response(response, output_format)
                    for s in samples:
                        f.write(json.dumps(s, ensure_ascii=False, default=str) + "\n")
                        generated += 1
                        if generated >= count:
                            break
                except Exception as e:
                    log(f"生成批次失敗: {e}")
                    time.sleep(1)
                    continue

        progress_cb(100, f"完成: {out_name} ({generated} 條)")
        log(f"AI 合成完成: {out_name} ({generated} 條)")
        return out_name

    def generate_preference_data(
        self,
        topic: str,
        count: int = 50,
        api_type: str = "ollama",
        api_base: str = "http://localhost:11434",
        api_key: str = "",
        model: str = "qwen2.5:7b",
        task: Optional[Task] = None,
    ) -> str:
        """生成 DPO/KTO 偏好數據"""
        progress_cb = task.update_progress if task else lambda p, m="": log(m)

        out_name = f"pref_{topic}_{count}_{get_timestamp()}.jsonl"
        out_path = DATASETS_DIR / out_name
        generated = 0

        system_prompt = (
            f"你是一個數據生成助手。為「{topic}」領域生成偏好訓練數據。\n"
            "每條數據包含：prompt（問題）、chosen（好回答）、rejected（差回答）。\n"
            "好回答應準確、詳細；差回答應有明顯缺陷（不完整、有錯誤或太簡短）。\n"
            "每條輸出一個 JSON 對象，格式：{\"prompt\":...,\"chosen\":...,\"rejected\":...}\n"
            "一行一條，不要加其他文字。"
        )

        with out_path.open("w", encoding="utf-8") as f:
            while generated < count:
                pct = (generated / count) * 90 + 5
                progress_cb(pct, f"生成偏好數據: {generated}/{count}")

                remaining = min(3, count - generated)
                user_prompt = f"請生成 {remaining} 條關於「{topic}」的偏好對比數據。"

                try:
                    response = self._call_llm(api_type, api_base, api_key, model, system_prompt, user_prompt)
                    for line in response.strip().split("\n"):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if all(k in obj for k in ("prompt", "chosen", "rejected")):
                                f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
                                generated += 1
                                if generated >= count:
                                    break
                        except json.JSONDecodeError:
                            continue
                except Exception as e:
                    log(f"偏好生成失敗: {e}")
                    time.sleep(1)

        progress_cb(100, f"完成: {out_name} ({generated} 條)")
        return out_name

    def _build_system_prompt(self, topic: str, style: str, difficulty: str, fmt: str) -> str:
        if fmt == "alpaca":
            format_desc = '每條格式：{"instruction":"指令","input":"可選輸入","output":"回答"}'
        elif fmt == "sharegpt":
            format_desc = '每條格式：{"conversations":[{"from":"human","value":"問題"},{"from":"gpt","value":"回答"}]}'
        else:
            format_desc = '每條格式：{"messages":[{"role":"user","content":"問題"},{"role":"assistant","content":"回答"}]}'

        return (
            f"你是專業的 AI 訓練數據生成器。\n"
            f"主題：{topic}\n"
            f"風格：{style}（informative=知識型 / creative=創意型 / technical=技術型）\n"
            f"難度：{difficulty}（easy/medium/hard）\n"
            f"輸出格式：{format_desc}\n"
            f"要求：\n"
            f"- 每行輸出一個 JSON 對象（JSONL 格式）\n"
            f"- 不要加 markdown 標記或其他裝飾文字\n"
            f"- 確保內容準確、多樣、有深度\n"
            f"- output/回答部分至少 50 字"
        )

    def _call_llm(self, api_type, api_base, api_key, model, system_prompt, user_prompt) -> str:
        """統一 LLM 調用"""
        if api_type == "ollama":
            return self._call_ollama(api_base, model, system_prompt, user_prompt)
        elif api_type == "openai":
            return self._call_openai(api_base, api_key, model, system_prompt, user_prompt)
        else:
            raise ValueError(f"不支持的 API 類型: {api_type}")

    def _call_ollama(self, api_base, model, system_prompt, user_prompt) -> str:
        """調用 Ollama API"""
        import urllib.request
        url = f"{api_base.rstrip('/')}/api/chat"
        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.8, "num_predict": 4096},
        }).encode("utf-8")

        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("message", {}).get("content", "")

    def _call_openai(self, api_base, api_key, model, system_prompt, user_prompt) -> str:
        """調用 OpenAI 兼容 API"""
        import urllib.request
        url = f"{api_base.rstrip('/')}/v1/chat/completions"
        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.8,
            "max_tokens": 4096,
        }).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = urllib.request.Request(url, data=payload, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]

    def _parse_response(self, response: str, fmt: str) -> List[Dict]:
        """解析 LLM 回覆為訓練樣本"""
        samples = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # 移除 markdown 代碼塊標記
            if line.startswith("```"):
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    samples.append(obj)
            except json.JSONDecodeError:
                # 嘗試從行中提取 JSON
                import re
                json_match = re.search(r'\{.*\}', line)
                if json_match:
                    try:
                        obj = json.loads(json_match.group())
                        samples.append(obj)
                    except json.JSONDecodeError:
                        pass
        return samples


ai_synthesizer = AISynthesizer()
