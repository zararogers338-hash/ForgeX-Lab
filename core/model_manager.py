# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

# core/model_manager.py - ForgeX v2.1
import json
from pathlib import Path
from typing import List, Dict, Optional
from core import MODELS_CACHE_DIR, LORAS_DIR, log, human_size, config
from core.task_queue import Task

RECOMMENDED_MODELS = [
    {"name":"Qwen/Qwen2.5-0.5B-Instruct","size":"0.5B","arch":"Qwen2","lang":"中/英","desc":"超輕量，8GB 入門首選"},
    {"name":"Qwen/Qwen2.5-1.5B-Instruct","size":"1.5B","arch":"Qwen2","lang":"中/英","desc":"輕量，RTX 5060 推薦"},
    {"name":"Qwen/Qwen2.5-3B-Instruct","size":"3B","arch":"Qwen2","lang":"中/英","desc":"平衡，8GB QLoRA 可訓"},
    {"name":"Qwen/Qwen2.5-7B-Instruct","size":"7B","arch":"Qwen2","lang":"中/英","desc":"主力，8GB QLoRA(seq≤1024)"},
    {"name":"Qwen/Qwen2.5-14B-Instruct","size":"14B","arch":"Qwen2","lang":"中/英","desc":"強力，需 16GB+"},
    {"name":"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B","size":"1.5B","arch":"DeepSeek","lang":"中/英","desc":"推理增強蒸餾"},
    {"name":"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B","size":"7B","arch":"DeepSeek","lang":"中/英","desc":"推理能力強"},
    {"name":"meta-llama/Llama-3.1-8B-Instruct","size":"8B","arch":"LLaMA","lang":"英","desc":"Meta 旗艦"},
    {"name":"google/gemma-2-2b-it","size":"2B","arch":"Gemma","lang":"英","desc":"Google 輕量"},
    {"name":"microsoft/Phi-3.5-mini-instruct","size":"3.8B","arch":"Phi","lang":"英","desc":"微軟小模型"},
    {"name":"mistralai/Mistral-7B-Instruct-v0.3","size":"7B","arch":"Mistral","lang":"英","desc":"高效架構"},
]

class ModelManager:
    def get_recommended_models(self) -> List[Dict]:
        return RECOMMENDED_MODELS

    def download_model(self, model_id: str, task=None) -> str:
        """下載 HF 模型 (FIX: 加上 task 參數)"""
        pcb = task.update_progress if task else lambda p, m="": log(m)
        pcb(5, f"開始下載: {model_id}")
        try:
            from huggingface_hub import snapshot_download
            import os
            if config.use_mirror:
                os.environ["HF_ENDPOINT"] = config.hf_mirror
            pcb(10, f"連接 {config.effective_hf_endpoint}...")
            path = snapshot_download(model_id, cache_dir=str(MODELS_CACHE_DIR),
                                     endpoint=config.effective_hf_endpoint)
            pcb(100, f"✅ {path}")
            return path
        except Exception as e:
            raise RuntimeError(f"下載失敗: {e}") from e

    def scan_local_models(self) -> List[Dict]:
        """快速扫描本地模型（项目缓存 + HF 默认缓存）"""
        models = []
        seen_names = set()
        import glob

        # 扫描目录列表：项目 cache + HF 默认 cache
        scan_dirs = [MODELS_CACHE_DIR]
        try:
            from pathlib import Path as _P
            hf_default = _P.home() / ".cache" / "huggingface" / "hub"
            if hf_default.exists() and hf_default != MODELS_CACHE_DIR:
                scan_dirs.append(hf_default)
            # Windows 下 HF_HOME 可能在别的位置
            import os
            hf_home = os.environ.get("HF_HOME")
            if hf_home:
                hf_hub = _P(hf_home) / "hub"
                if hf_hub.exists() and hf_hub not in scan_dirs:
                    scan_dirs.append(hf_hub)
        except Exception:
            pass

        for cache_dir in scan_dirs:
            try:
                for d in glob.glob(str(cache_dir / "models--*")):
                    dp = Path(d)
                    name = dp.name.replace("models--", "").replace("--", "/")
                    if name in seen_names:
                        continue
                    snap = dp / "snapshots"
                    if snap.exists():
                        for s in snap.iterdir():
                            if s.is_dir() and (s / "config.json").exists():
                                models.append({"name": name, "path": str(s), "source": "huggingface"})
                                seen_names.add(name)
                                break
            except Exception:
                continue

        if LORAS_DIR.exists():
            for d in LORAS_DIR.iterdir():
                if d.is_dir() and (d/"config.json").exists() and not (d/"adapter_config.json").exists():
                    models.append({"name": d.name, "path": str(d), "source": "local"})
        return models

    def scan_all_available(self) -> List[Dict]:
        items = self.scan_local_models()

        # include gguf under data/ and project (user exports)
        try:
            from core.config import PROJECT_DIR, DATA_DIR
            roots = [DATA_DIR, PROJECT_DIR]
            seen = set()
            for r in roots:
                for p in Path(r).rglob("*.gguf"):
                    sp = str(p.resolve())
                    if sp in seen:
                        continue
                    seen.add(sp)
                    items.append({"name": p.name, "path": sp, "source": "gguf"})
        except Exception:
            pass

        from core.merger import merger
        for l in merger.list_available_loras():
            items.append({"name": l["filename"], "path": l["path"], "source": "lora"})
        return items
model_manager = ModelManager()
