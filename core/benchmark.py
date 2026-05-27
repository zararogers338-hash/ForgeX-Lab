# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

# core/benchmark.py - ForgeX v2.1
import json, time
from typing import Dict, List, Optional
from pathlib import Path
from core import LORAS_DIR, log
from core.task_queue import Task

class Benchmark:
    DIMS = ["reasoning","creativity","knowledge","instruction_following","coding"]
    TESTS = {
        "reasoning": [
            {"prompt":"A管每小時進水3噸，B管每小時進水2噸，C管每小時放水4噸。水池20噸，三管全開多久裝滿？","criteria":"20小時","answer_key":"20"},
            {"prompt":"一個房間有3盞燈和3個開關在門外，只能進入一次。怎麼確定對應？","criteria":"利用溫度","answer_key":"溫度"},
        ],
        "creativity": [
            {"prompt":"用100字描繪賽博龐克風格的城市雨夜。","criteria":"畫面感、修辭","answer_key":None},
        ],
        "knowledge": [
            {"prompt":"簡述量子纠缠的原理及在量子計算中的作用。","criteria":"準確描述","answer_key":None},
        ],
        "instruction_following": [
            {"prompt":"用JSON格式列出3種排序算法，每個包含name和complexity。","criteria":"JSON格式","answer_key":"json"},
        ],
        "coding": [
            {"prompt":"用Python寫斐波那契第n項函數，用動態規劃。","criteria":"語法正確、DP","answer_key":"def "},
        ],
    }

    def __init__(self):
        self._pipe = None

    def _get_pipe(self, mp):
        if self._pipe and getattr(self._pipe, '_mp', None) == str(mp):
            return self._pipe
        from transformers import pipeline
        import torch
        self._pipe = pipeline("text-generation", model=str(mp),
                              torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        self._pipe._mp = str(mp)
        return self._pipe


    def _get_gguf(self, mp: Path):
        if getattr(self, "_gguf", None) is not None and getattr(self._gguf, "_mp", None) == str(mp):
            return self._gguf
        from llama_cpp import Llama
        llm = Llama(model_path=str(mp), n_ctx=2048, n_gpu_layers=-1, verbose=False)
        llm._mp = str(mp)
        self._gguf = llm
        return llm

    def _gen_gguf(self, llm, prompt, max_new=512):
        try:
            out = llm(prompt, max_tokens=int(max_new), temperature=0.7, top_p=0.9, stop=["</s>", "<|end|>", "<|im_end|>"])
            return ((out.get("choices") or [{}])[0].get("text","") or "").strip()
        except Exception as e:
            return f"[Error: {e}]"
    def _gen(self, pipe, prompt, max_new=512):
        try:
            out = pipe(prompt, max_new_tokens=max_new, do_sample=True, temperature=0.7, top_p=0.9, return_full_text=False)
            return out[0]["generated_text"].strip() if out else ""
        except Exception as e:
            return f"[Error: {e}]"

    def _score(self, ans, test):
        if not ans or ans.startswith("[Error"): return 0
        s = 2 if len(ans) > 20 else 0
        if len(ans) > 100: s += 1
        ak = test.get("answer_key")
        if ak:
            s += 5 if ak.lower() in ans.lower() else 1
        else:
            s += 3 if len(ans) > 50 else 1
        return min(s, 10)

    def quick_test(self, model_path, task=None):
        pcb = task.update_progress if task else lambda p, m="": log(m)
        pcb(5, "載入模型...")
        mp = Path(model_path)
        is_gguf = str(mp).lower().endswith('.gguf')
        if is_gguf:
            pcb(8, '載入 GGUF...')
            llm = self._get_gguf(mp)
        else:
            pipe = self._get_pipe(mp)

        results = {}; total = 0; count = 0; done = 0
        tests = {d: self.TESTS[d][:1] for d in self.DIMS}
        n = sum(len(v) for v in tests.values())
        for dim, tl in tests.items():
            ds = []
            for t in tl:
                done += 1; pcb(10+80*done//n, f"測試 {dim}...")
                a = (self._gen_gguf(llm, t["prompt"]) if is_gguf else self._gen(pipe, t["prompt"])); s = self._score(a, t)
                ds.append({"prompt": t["prompt"][:50], "score": s, "preview": a[:200]})
                total += s; count += 1
            results[dim] = {"scores": ds, "avg": round(sum(x["score"] for x in ds)/len(ds), 1)}
        avg = round(total/max(count,1), 1)
        rpt = f"📊 快速測試 — {Path(model_path).name}\n{'='*50}\n"
        for d in self.DIMS:
            if d in results:
                rpt += f"\n【{d}】{results[d]['avg']}/10\n"
                for r in results[d]["scores"]:
                    rpt += f"  • {r['prompt']}... → {r['score']}/10\n"
        rpt += f"\n{'='*50}\n綜合: {avg}/10\n"
        pcb(100, f"完成 {avg}/10")
        return {"report": rpt, "scores": results, "avg": avg}

    def full_test(self, model_path, task=None):
        pcb = task.update_progress if task else lambda p, m="": log(m)
        pcb(5, "載入模型...")
        mp = Path(model_path)
        is_gguf = str(mp).lower().endswith('.gguf')
        if is_gguf:
            pcb(8, '載入 GGUF...')
            llm = self._get_gguf(mp)
        else:
            pipe = self._get_pipe(mp)

        results = {}; total = 0; count = 0; done = 0
        n = sum(len(v) for v in self.TESTS.values())
        for dim, tl in self.TESTS.items():
            ds = []
            for t in tl:
                done += 1; pcb(10+80*done//n, f"[{done}/{n}] {dim}...")
                a = (self._gen_gguf(llm, t["prompt"]) if is_gguf else self._gen(pipe, t["prompt"])); s = self._score(a, t)
                ds.append({"prompt": t["prompt"][:50], "score": s, "preview": a[:300]})
                total += s; count += 1
            results[dim] = {"scores": ds, "avg": round(sum(x["score"] for x in ds)/len(ds), 1)}
        avg = round(total/max(count,1), 1)
        rpt = f"📊 完整測試 — {Path(model_path).name}\n{'='*60}\n"
        for d in self.DIMS:
            if d in results:
                rpt += f"\n【{d}】{results[d]['avg']}/10\n"
                for r in results[d]["scores"]:
                    rpt += f"  • {r['prompt']}... → {r['score']}/10\n"
        rpt += f"\n{'='*60}\n綜合: {avg}/10 ({count}題)\n"
        pcb(100, f"完成 {avg}/10")
        return {"report": rpt, "scores": results, "avg": avg}

benchmark = Benchmark()
