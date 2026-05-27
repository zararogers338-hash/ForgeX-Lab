# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

# core/dataset_manager.py - ForgeX v2 數據集管理器
import json
import csv
import random
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any
from collections import Counter

from core import DATASETS_DIR, log, safe_json_load, safe_json_save, get_timestamp, random_name, human_size


class DatasetManager:
    """數據集管理 - 上傳/列出/預覽/刪除/格式轉換/統計/清洗"""

    SUPPORTED_FORMATS = {".jsonl", ".json", ".csv", ".txt", ".parquet"}
    TRAIN_FORMATS = {"alpaca", "sharegpt", "openai", "text"}

    def __init__(self):
        self.index_path = DATASETS_DIR / "index.json"
        self.datasets: List[Dict] = self._load_index()

    def _load_index(self) -> List[Dict]:
        index = safe_json_load(self.index_path, [])
        valid = []
        for entry in index:
            path = DATASETS_DIR / entry.get("filename", "")
            if path.exists():
                valid.append(entry)
        if len(valid) != len(index):
            safe_json_save(self.index_path, valid)
        return valid

    def _save_index(self):
        safe_json_save(self.index_path, self.datasets)

    # ============================ CRUD ============================
    def upload(self, file_path: Path, *, original_name: str | None = None) -> Dict:
        """上傳數據集

        設計原則：
        - 不在檔名上追加日期/時間戳（用戶感知的檔名應保持穩定）。
        - 同名上傳視為覆蓋更新（更新 index 與統計）。
        """
        suffix = file_path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"不支持的格式: {suffix}（支持: {', '.join(self.SUPPORTED_FORMATS)}）")

        # Gradio 上傳時 file_path 往往是 temp 路徑，可能丟失原始檔名。
        # 允許 UI 傳入 original_name 作為最終檔名。
        final_name = (original_name or file_path.name)
        final_name = Path(final_name).name  # drop any dirs
        if not Path(final_name).suffix:
            # 若沒有副檔名，補回來（避免無法讀取/格式偵測失敗）
            final_name = f"{final_name}{suffix}"
        new_path = DATASETS_DIR / final_name

        # 覆蓋寫入，不做 timestamp rename。
        try:
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, new_path)
        except Exception as e:
            raise RuntimeError(f"上傳失敗（無法寫入 {new_path}）：{e}")

        log(f"上傳數據集: {new_path.name}（覆蓋更新）" if new_path.exists() else f"上傳數據集: {new_path.name}")

        stats = self._compute_stats(new_path)
        detected_format = self._detect_format(new_path)

        # 更新 index：同名 -> 覆蓋更新，否則新增
        entry = None
        for d in self.datasets:
            if d.get("filename") == new_path.name:
                entry = d
                break

        if entry is None:
            entry = {"filename": new_path.name}
            self.datasets.append(entry)

        entry.update({
            "size": new_path.stat().st_size,
            "samples": stats.get("count", -1),
            "uploaded_at": get_timestamp(),
            "format_type": suffix,
            "detected_train_format": detected_format,
            "stats": stats,
        })
        self._save_index()
        return entry

    def list_datasets(self) -> List[Dict]:
        return self.datasets

    def delete(self, filename: str) -> bool:
        path = DATASETS_DIR / filename
        if path.exists():
            path.unlink()
        self.datasets = [d for d in self.datasets if d["filename"] != filename]
        self._save_index()
        log(f"刪除數據集: {filename}")
        return True

    def preview(self, filename: str, n: int = 5, *, max_str: int = 1200, max_keys: int = 60) -> List[Dict]:
        """預覽前 n 條（做截斷，避免 UI 因超長字段卡死）"""
        path = DATASETS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"數據集不存在: {filename}")

        def trunc(obj):
            # Keep it simple: strings trimmed, dict keys limited, lists limited.
            if isinstance(obj, str):
                return obj if len(obj) <= max_str else obj[:max_str] + "…(truncated)"
            if isinstance(obj, (int, float, bool)) or obj is None:
                return obj
            if isinstance(obj, list):
                out = [trunc(x) for x in obj[:20]]
                if len(obj) > 20:
                    out.append(f"…(+{len(obj)-20} items)")
                return out
            if isinstance(obj, dict):
                out = {}
                for i, (k, v) in enumerate(obj.items()):
                    if i >= max_keys:
                        out["…"] = f"(+{len(obj)-max_keys} keys)"
                        break
                    out[str(k)] = trunc(v)
                return out
            try:
                return trunc(str(obj))
            except Exception:
                return "<unserializable>"

        suffix = path.suffix.lower()
        samples: List[Dict] = []

        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    if i >= n:
                        break
                    line = line.strip()
                    if line:
                        try:
                            samples.append(trunc(json.loads(line)))
                        except json.JSONDecodeError:
                            samples.append({"_raw": trunc(line)})

        elif suffix == ".json":
            data = safe_json_load(path, [])
            if isinstance(data, list):
                samples = [trunc(x) for x in data[:n]]
            elif isinstance(data, dict):
                samples = [trunc(data)]

        elif suffix == ".csv":
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i >= n:
                        break
                    samples.append(trunc(dict(row)))

        elif suffix == ".txt":
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    if i >= n:
                        break
                    samples.append({"text": trunc(line.strip())})

        else:
            # parquet or unknown: best-effort preview
            if suffix == ".parquet":
                try:
                    import pandas as pd
                    df = pd.read_parquet(path, engine="auto")
                    for _, row in df.head(n).iterrows():
                        samples.append(trunc(row.to_dict()))
                except Exception as e:
                    samples = [{"_error": f"Parquet 预览失败: {e}"}]
            else:
                with path.open("rb") as f:
                    head = f.read(2048)
                samples = [{"_raw_bytes_head": head.hex()[:max_str]}]

        return samples

    # ============================ 格式檢測 ============================
    def _detect_format(self, path: Path) -> str:
        """自動檢測訓練數據格式"""
        try:
            samples = self.preview(path.name, n=3)
            if not samples:
                return "unknown"

            sample = samples[0]
            keys = set(sample.keys()) if isinstance(sample, dict) else set()

            # Alpaca 格式
            if {"instruction", "output"}.issubset(keys):
                return "alpaca"
            # ShareGPT 格式
            if "conversations" in keys or "messages" in keys:
                return "sharegpt"
            # OpenAI 格式
            if "messages" in keys and isinstance(sample.get("messages"), list):
                msgs = sample["messages"]
                if msgs and isinstance(msgs[0], dict) and "role" in msgs[0]:
                    return "openai"
            # 純文本
            if "text" in keys and len(keys) <= 2:
                return "text"

            return "unknown"
        except Exception:
            return "unknown"

    # ============================ 統計 ============================
    def _compute_stats(self, path: Path) -> Dict:
        """計算數據集統計信息"""
        suffix = path.suffix.lower()
        stats = {"count": 0, "fields": [], "avg_length": 0}

        try:
            if suffix == ".jsonl":
                lengths = []
                field_counter = Counter()
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        stats["count"] += 1
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict):
                                field_counter.update(obj.keys())
                                lengths.append(len(line))
                        except json.JSONDecodeError:
                            pass
                stats["fields"] = [k for k, _ in field_counter.most_common(20)]
                stats["avg_length"] = round(sum(lengths) / len(lengths)) if lengths else 0

            elif suffix == ".json":
                data = safe_json_load(path, [])
                if isinstance(data, list):
                    stats["count"] = len(data)
                    if data and isinstance(data[0], dict):
                        stats["fields"] = list(data[0].keys())
                elif isinstance(data, dict):
                    stats["count"] = 1
                    stats["fields"] = list(data.keys())

            elif suffix == ".csv":
                with path.open("r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader, [])
                    stats["fields"] = header
                    stats["count"] = sum(1 for _ in reader)

            elif suffix == ".txt":
                with path.open("r", encoding="utf-8") as f:
                    stats["count"] = sum(1 for line in f if line.strip())

            elif suffix == ".parquet":
                try:
                    import pandas as pd
                    df = pd.read_parquet(path, engine="auto")
                    stats["count"] = len(df)
                    stats["fields"] = list(df.columns)
                except Exception as e:
                    log(f"Parquet 統計失敗 {path.name}: {e}")

        except Exception as e:
            log(f"統計數據集失敗 {path.name}: {e}")

        return stats

    # ============================ 格式轉換 ============================
    def convert_format(self, filename: str, target_format: str) -> str:
        """轉換數據格式：alpaca ↔ sharegpt ↔ openai"""
        if target_format not in self.TRAIN_FORMATS:
            raise ValueError(f"目標格式不支持: {target_format}")

        path = DATASETS_DIR / filename
        samples = []

        # 讀取所有數據
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not samples:
            raise ValueError("數據集為空")

        source_format = self._detect_format(path)
        converted = []

        for sample in samples:
            try:
                converted.append(self._convert_sample(sample, source_format, target_format))
            except Exception:
                continue

        # 保存
        out_name = f"{path.stem}_{target_format}_{get_timestamp()}.jsonl"
        out_path = DATASETS_DIR / out_name
        with out_path.open("w", encoding="utf-8") as f:
            for item in converted:
                f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

        # 注冊到索引
        entry = {
            "filename": out_name,
            "size": out_path.stat().st_size,
            "samples": len(converted),
            "uploaded_at": get_timestamp(),
            "format_type": ".jsonl",
            "detected_train_format": target_format,
            "converted_from": filename,
        }
        self.datasets.append(entry)
        self._save_index()
        log(f"格式轉換完成: {filename} → {out_name} ({len(converted)} 條)")
        return out_name

    def _convert_sample(self, sample: dict, source: str, target: str) -> dict:
        """單條數據格式轉換"""
        # 先統一轉為 messages 格式
        messages = []

        if source == "alpaca":
            user_content = sample.get("instruction", "")
            if sample.get("input"):
                user_content += f"\n{sample['input']}"
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": sample.get("output", "")},
            ]
            if sample.get("system"):
                messages.insert(0, {"role": "system", "content": sample["system"]})

        elif source in ("sharegpt", "openai"):
            convos = sample.get("conversations") or sample.get("messages", [])
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            for msg in convos:
                role = role_map.get(msg.get("from") or msg.get("role"), msg.get("from") or msg.get("role"))
                content = msg.get("value") or msg.get("content", "")
                messages.append({"role": role, "content": content})

        elif source == "text":
            messages = [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": sample.get("text", str(sample))},
            ]

        # 從 messages 轉為目標格式
        if target == "alpaca":
            user_msgs = [m for m in messages if m["role"] == "user"]
            asst_msgs = [m for m in messages if m["role"] == "assistant"]
            sys_msgs = [m for m in messages if m["role"] == "system"]
            return {
                "instruction": user_msgs[0]["content"] if user_msgs else "",
                "input": "",
                "output": asst_msgs[0]["content"] if asst_msgs else "",
                **({"system": sys_msgs[0]["content"]} if sys_msgs else {}),
            }

        elif target == "sharegpt":
            role_map = {"user": "human", "assistant": "gpt", "system": "system"}
            return {
                "conversations": [
                    {"from": role_map.get(m["role"], m["role"]), "value": m["content"]}
                    for m in messages
                ]
            }

        elif target == "openai":
            return {"messages": messages}

        elif target == "text":
            text_parts = []
            for m in messages:
                if m["role"] == "user":
                    text_parts.append(f"### Human: {m['content']}")
                elif m["role"] == "assistant":
                    text_parts.append(f"### Assistant: {m['content']}")
            return {"text": "\n\n".join(text_parts)}

        return sample

    # ============================ 數據清洗 ============================
    def clean_dataset(self, filename: str, options: Dict) -> str:
        """數據清洗"""
        path = DATASETS_DIR / filename
        samples = []

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        original_count = len(samples)

        # 去重
        if options.get("deduplicate", False):
            seen = set()
            unique = []
            for s in samples:
                key = json.dumps(s, sort_keys=True, ensure_ascii=False, default=str)
                if key not in seen:
                    seen.add(key)
                    unique.append(s)
            samples = unique

        # 過濾短樣本
        min_length = options.get("min_length", 0)
        if min_length > 0:
            samples = [s for s in samples if len(json.dumps(s, ensure_ascii=False, default=str)) >= min_length]

        # 過濾空字段
        if options.get("remove_empty", False):
            def has_content(s):
                if isinstance(s, dict):
                    return any(bool(str(v).strip()) for v in s.values())
                return bool(str(s).strip())
            samples = [s for s in samples if has_content(s)]

        # 保存
        out_name = f"{path.stem}_cleaned_{get_timestamp()}.jsonl"
        out_path = DATASETS_DIR / out_name
        with out_path.open("w", encoding="utf-8") as f:
            for item in samples:
                f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

        entry = {
            "filename": out_name,
            "size": out_path.stat().st_size,
            "samples": len(samples),
            "uploaded_at": get_timestamp(),
            "format_type": ".jsonl",
            "detected_train_format": self._detect_format(out_path),
            "cleaned_from": filename,
            "removed": original_count - len(samples),
        }
        self.datasets.append(entry)
        self._save_index()
        log(f"清洗完成: {filename} → {out_name} (移除 {original_count - len(samples)} 條)")
        return out_name

    # ============================ 分割 ============================
    def split_dataset(self, filename: str, train_ratio: float = 0.9) -> Dict[str, str]:
        """分割訓練/驗證集"""
        path = DATASETS_DIR / filename
        samples = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        random.shuffle(samples)
        split_idx = int(len(samples) * train_ratio)
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]

        result = {}
        for subset_name, subset_data in [("train", train_samples), ("val", val_samples)]:
            out_name = f"{path.stem}_{subset_name}_{get_timestamp()}.jsonl"
            out_path = DATASETS_DIR / out_name
            with out_path.open("w", encoding="utf-8") as f:
                for item in subset_data:
                    f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

            entry = {
                "filename": out_name,
                "size": out_path.stat().st_size,
                "samples": len(subset_data),
                "uploaded_at": get_timestamp(),
                "format_type": ".jsonl",
                "split_from": filename,
                "split_type": subset_name,
            }
            self.datasets.append(entry)
            result[subset_name] = out_name

        self._save_index()
        log(f"分割完成: {filename} → train({len(train_samples)}) + val({len(val_samples)})")
        return result

    # ============================ AI 合成（v2 重寫）============================
    def generate_synthetic(self, topic: str, count: int = 100, fmt: str = "alpaca") -> str:
        """生成合成數據集（模板 + 變異，v2 改進版）"""
        if fmt not in ("alpaca", "sharegpt"):
            raise ValueError("格式僅支持 alpaca / sharegpt")

        out_name = f"synthetic_{topic}_{count}_{get_timestamp()}.jsonl"
        out_path = DATASETS_DIR / out_name

        # 豐富的模板庫
        alpaca_templates = [
            {"instruction": "解釋{topic}的基本概念", "input": "", "output": ""},
            {"instruction": "舉一個{topic}的實際應用例子", "input": "", "output": ""},
            {"instruction": "{topic}的優缺點是什麼？", "input": "", "output": ""},
            {"instruction": "比較{topic}和相關領域的異同", "input": "", "output": ""},
            {"instruction": "簡述{topic}的發展歷史", "input": "", "output": ""},
            {"instruction": "初學者應該如何學習{topic}？", "input": "", "output": ""},
            {"instruction": "{topic}中最常見的誤解是什麼？", "input": "", "output": ""},
            {"instruction": "在{topic}領域中，什麼是最新的趨勢？", "input": "", "output": ""},
            {"instruction": "請用簡單的語言解釋{topic}給一個10歲的孩子聽", "input": "", "output": ""},
            {"instruction": "{topic}的核心原理是什麼？", "input": "", "output": ""},
        ]

        sharegpt_templates = [
            {"conversations": [
                {"from": "human", "value": "告訴我關於{topic}的基礎知識"},
                {"from": "gpt", "value": ""},
            ]},
            {"conversations": [
                {"from": "human", "value": "{topic}是什麼？為什麼重要？"},
                {"from": "gpt", "value": ""},
            ]},
            {"conversations": [
                {"from": "human", "value": "你能舉個{topic}的例子嗎？"},
                {"from": "gpt", "value": ""},
                {"from": "human", "value": "能再詳細解釋一下嗎？"},
                {"from": "gpt", "value": ""},
            ]},
        ]

        templates = alpaca_templates if fmt == "alpaca" else sharegpt_templates

        with out_path.open("w", encoding="utf-8") as f:
            for _ in range(count):
                tmpl = random.choice(templates)
                # 安全的變異：只替換佔位符，不破壞 JSON 結構
                item_str = json.dumps(tmpl, ensure_ascii=False, default=str)
                item_str = item_str.replace("{topic}", topic)
                f.write(item_str + "\n")

        entry = {
            "filename": out_name,
            "size": out_path.stat().st_size,
            "samples": count,
            "uploaded_at": get_timestamp(),
            "format_type": ".jsonl",
            "detected_train_format": fmt,
            "synthetic": True,
            "topic": topic,
        }
        self.datasets.append(entry)
        self._save_index()
        log(f"合成數據集完成: {out_name} ({count} 條)")
        return out_name


    # ============================ 清洗 ============================
    def deduplicate(self, filename: str) -> str:
        """Best-effort deduplication.

        - jsonl/txt: remove exact duplicate lines (stable order).
        - json(list): remove exact duplicate json-serialized items.
        - csv: remove exact duplicate rows (including header preserved).
        Output overwrites the same filename (in-place) and refreshes stats/index.
        """
        path = DATASETS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(filename)

        suffix = path.suffix.lower()
        tmp = path.with_suffix(path.suffix + ".dedup.tmp")

        if suffix in [".jsonl", ".txt"]:
            seen = set()
            with path.open("r", encoding="utf-8", errors="ignore") as fin, tmp.open("w", encoding="utf-8") as fout:
                for line in fin:
                    l = line.rstrip("\n")
                    if not l.strip():
                        continue
                    if l in seen:
                        continue
                    seen.add(l)
                    fout.write(l + "\n")
            tmp.replace(path)

        elif suffix == ".json":
            data = safe_json_load(path, [])
            if isinstance(data, list):
                seen = set()
                out = []
                for obj in data:
                    key = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(obj)
                safe_json_save(path, out)

        elif suffix == ".csv":
            with path.open("r", encoding="utf-8", errors="ignore", newline="") as fin, tmp.open("w", encoding="utf-8", newline="") as fout:
                reader = csv.reader(fin)
                writer = csv.writer(fout)
                header = next(reader, None)
                if header:
                    writer.writerow(header)
                seen = set()
                for row in reader:
                    key = "\t".join(row)
                    if key in seen:
                        continue
                    seen.add(key)
                    writer.writerow(row)
            tmp.replace(path)

        else:
            raise ValueError(f"不支持此格式的去重: {suffix}")

        # refresh stats in index
        for d in self.datasets:
            if d.get("filename") == filename:
                d["stats"] = self._compute_stats(path)
                break
        self._save_index()
        return filename

# 單例
dataset_manager = DatasetManager()


