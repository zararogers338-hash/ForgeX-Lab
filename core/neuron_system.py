# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

# core/neuron_system.py - 差分神经元系统 (统一版)
#
# 两种模式, 一套接口:
#
#   🔬 MoLoRA 模式 (单模型多专家)
#     一个基座 + N 组 LoRA 专家 + 内置门控
#     训练后合并 → 标准模型, 兼容所有推理软件
#     适合: 训练阶段, 想要一个模型文件搞定
#
#   🐝 蜂群模式 (多模型路由)
#     小门控模型 + N 个独立专家模型
#     显存 = 门控 + 1 个专家, 按需换入换出
#     适合: 已有多个专业模型, 想让它们协同工作
#
# 共享:
#   - NeuronDef (神经元/专家定义)
#   - 关键词路由 + 领域关键词库
#   - 路由预览
#   - 配置持久化

import gc
import json
import re
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from core.logger import log

CONFIG_DIR = Path("data/neuron_configs")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════
# 常量 / 预置数据
# ═══════════════════════════════════════════════════

DOMAIN_CHOICES = ["医学", "法律", "代码", "金融", "学术", "通用", "自定义"]
NEURON_COLORS = [
    "#3B82F6", "#EF4444", "#10B981", "#F59E0B",
    "#8B5CF6", "#EC4899", "#14B8A6", "#F97316",
]

DOMAIN_KEYWORDS = {
    "医学": ["诊断", "症状", "治疗", "药物", "临床", "患者", "病理", "处方",
             "手术", "检查", "病例", "医生", "护理", "疾病", "感染", "肿瘤",
             "血压", "心率", "过敏", "疫苗", "抗生素", "CT", "MRI", "B超"],
    "法律": ["法律", "合同", "诉讼", "仲裁", "判决", "侵权", "专利", "版权",
             "劳动法", "民法", "刑法", "赔偿", "违约", "起诉", "辩护", "证据",
             "法规", "条款", "协议", "担保", "抵押", "继承", "离婚"],
    "代码": ["代码", "函数", "API", "Python", "Java", "bug", "debug", "算法",
             "编程", "变量", "循环", "数组", "数据库", "SQL", "git", "docker",
             "import", "class", "def", "return", "error", "exception", "compile"],
    "金融": ["股票", "基金", "利率", "通胀", "GDP", "投资", "理财", "债券",
             "外汇", "期货", "期权", "风险", "收益", "资产", "负债", "估值"],
    "学术": ["论文", "研究", "实验", "假设", "数据", "统计", "分析", "文献",
             "引用", "摘要", "方法论", "样本", "变量", "显著性", "peer review"],
}


# ═══════════════════════════════════════════════════
# 数据结构 (统一)
# ═══════════════════════════════════════════════════

@dataclass
class NeuronDef:
    """一个神经元定义 — 两种模式通用"""
    name: str                                   # "医学专家"
    domain: str = "通用"                         # "医学" / "法律" ...
    description: str = ""                       # 系统提示词
    keywords: List[str] = field(default_factory=list)
    priority: int = 5                           # 0-10
    color: str = "#3B82F6"
    enabled: bool = True
    # MoLoRA 模式字段
    datasets: List[str] = field(default_factory=list)  # 训练数据
    lora_path: str = ""                         # 已有 LoRA (初始化用)
    # 蜂群模式字段
    model_path: str = ""                        # HF ID 或本地路径
    quantize_4bit: bool = True
    # 统计
    total_calls: int = 0
    avg_latency_ms: float = 0.0

    def to_dict(self): return asdict(self)

    @classmethod
    def from_dict(cls, d):
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass
class NeuronConfig:
    """统一配置"""
    name: str = "default"
    mode: str = "swarm"                         # "molora" | "swarm"
    neurons: List[NeuronDef] = field(default_factory=list)

    # ── MoLoRA 模式参数 ──
    base_model: str = ""
    rank: int = 16
    top_k: int = 2
    alpha: float = 32.0
    lr: float = 2e-4
    epochs: float = 1.0
    max_seq_len: int = 2048
    aux_loss_weight: float = 0.01

    # ── 蜂群模式参数 ──
    gateway_model: str = ""                     # 门控小模型
    gateway_quantize: bool = False
    route_mode: str = "hybrid"                  # gateway_llm / keyword / hybrid
    cache_in_cpu: bool = True
    max_cpu_cache: int = 1

    def to_dict(self):
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["neurons"] = [n.to_dict() for n in self.neurons]
        return d

    @classmethod
    def from_dict(cls, d):
        neurons = [NeuronDef.from_dict(n) for n in d.pop("neurons", [])]
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        cfg = cls(**{k: v for k, v in d.items() if k in valid})
        cfg.neurons = neurons
        return cfg


# ═══════════════════════════════════════════════════
# 关键词路由 (共享)
# ═══════════════════════════════════════════════════

def keyword_route(query: str, neurons: List[NeuronDef]) -> List[Tuple[NeuronDef, float]]:
    """关键词匹配路由 — 毫秒级, 不需要模型"""
    scores = []
    q = query.lower()
    for n in neurons:
        if not n.enabled:
            continue
        score = 0.0
        for kw in n.keywords:
            if kw.lower() in q:
                score += 3.0
        for kw in DOMAIN_KEYWORDS.get(n.domain, []):
            if kw.lower() in q:
                score += 1.0
        score += n.priority * 0.1
        if score < 0.1:
            score = 0.1
        scores.append((n, score))
    scores.sort(key=lambda x: -x[1])
    return scores


def preview_routing(query: str, neurons: List[NeuronDef]) -> List[Dict]:
    """路由预览 (UI 用)"""
    scores = keyword_route(query, neurons)
    results = []
    q = query.lower()
    for n, score in scores:
        hits = [kw for kw in n.keywords if kw.lower() in q]
        domain_hits = [kw for kw in DOMAIN_KEYWORDS.get(n.domain, []) if kw.lower() in q]
        results.append({
            "name": n.name, "domain": n.domain, "score": score,
            "matched": hits + domain_hits,
            "model_path": n.model_path,
        })
    return results


def format_route_preview(query: str, neurons: List[NeuronDef], mode: str = "swarm") -> str:
    """格式化路由预览为 Markdown"""
    results = preview_routing(query, neurons)
    if not results:
        return "❌ 无可用神经元"

    max_s = max(r["score"] for r in results) if results else 1
    lines = [f"### 🔍 路由预览\n**查询**: {query}\n"]
    for i, r in enumerate(results[:5]):
        medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
        pct = r["score"] / max_s * 100 if max_s > 0 else 0
        bar = "█" * int(pct / 2.5) + "░" * (40 - int(pct / 2.5))
        lines.append(f"{medal} **{r['name']}** [{r['domain']}] — 得分: **{r['score']:.1f}**")
        lines.append(f"  `{bar}`")
        if mode == "swarm" and r["model_path"]:
            lines.append(f"  模型: `{r['model_path']}`")
        if r["matched"]:
            lines.append(f"  命中: {', '.join(r['matched'][:8])}")
        lines.append("")

    lines.append(f"---\n**将激活**: 🧠 {results[0]['name']}")
    if mode == "swarm":
        lines.append("\n> 实际推理时门控 LLM 做最终决策 (此处仅关键词预览)")
    else:
        lines.append("\n> 实际路由由 MoLoRA 内部门控决定 (此处仅关键词预览)")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════
# 配置持久化
# ═══════════════════════════════════════════════════

def save_config(config: NeuronConfig) -> str:
    path = CONFIG_DIR / f"{config.name}.json"
    path.write_text(json.dumps(config.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"💾 神经元配置已保存: {path}")
    return str(path)


def load_config(name: str) -> Optional[NeuronConfig]:
    path = CONFIG_DIR / f"{name}.json"
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return NeuronConfig.from_dict(d)
    except Exception as e:
        log(f"⚠️ 加载配置失败 {path}: {e}")
        return None


def list_configs() -> List[str]:
    return sorted(p.stem for p in CONFIG_DIR.glob("*.json"))


# ═══════════════════════════════════════════════════
# MoLoRA 模式: 训练
# ═══════════════════════════════════════════════════

def train_molora(config: NeuronConfig, task=None):
    """用 MoLoRA 训练多专家模型 (单基座)"""
    from core.trainer import Trainer, _safe_update

    if config.mode != "molora":
        raise ValueError("当前配置不是 MoLoRA 模式")

    # 收集所有神经元的数据集
    all_datasets = []
    labels = []
    for n in config.neurons:
        all_datasets.extend(n.datasets)
        labels.append(n.name)

    if not all_datasets:
        raise RuntimeError("无训练数据集（请为神经元指定数据集）")

    trainer = Trainer()
    params = {
        "output_name": f"molora_{config.name}",
        "lr": config.lr,
        "batch_size": 1,
        "epochs": config.epochs,
        "max_seq_len": config.max_seq_len,
        "use_qlora": True,
        "rank": config.rank,
        "gradient_accumulation_steps": 4,
        # MoLoRA 专用
        "use_molora": True,
        "molora_n_experts": len(config.neurons),
        "molora_top_k": config.top_k,
        "molora_expert_labels": labels,
        "molora_aux_weight": config.aux_loss_weight,
    }
    return trainer.train(
        method="SFT", backend="trl",
        base_model=config.base_model,
        dataset_path=list(set(all_datasets)),
        params=params, task=task,
    )


# ═══════════════════════════════════════════════════
# 蜂群模式: 门控路由器
# ═══════════════════════════════════════════════════

def _build_gateway_prompt(query: str, neurons: List[NeuronDef]) -> str:
    expert_desc = "\n".join(
        f"  {i+1}. [{n.name}] 领域: {n.domain} — {n.description or '通用'}"
        for i, n in enumerate(neurons) if n.enabled
    )
    return (
        "你是一个智能路由器。根据用户的问题，选择最合适的专家来回答。\n"
        "只回复专家编号（数字），不要解释。\n\n"
        f"可用专家:\n{expert_desc}\n\n"
        f"用户问题: {query}\n\n"
        "最合适的专家编号是: "
    )


class GatewayRouter:
    """门控路由器 — 小 LLM 做路由判断"""
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_path = None
        self._lock = threading.Lock()

    def load(self, model_path: str, quantize: bool = False):
        if self.model_path == model_path and self.model is not None:
            return
        self.unload()
        log(f"🚪 加载门控: {model_path}")
        t0 = time.time()
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        kw = {"device_map": "auto", "trust_remote_code": True}
        if quantize:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        else:
            kw["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
        self.tokenizer = tok
        self.model_path = model_path
        log(f"🚪 门控就绪 ({time.time()-t0:.1f}s)")

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = self.tokenizer = self.model_path = None
            gc.collect()
            try:
                import torch; torch.cuda.empty_cache()
            except Exception: pass

    def route(self, query: str, neurons: List[NeuronDef]) -> List[Tuple[NeuronDef, float]]:
        import torch
        if self.model is None:
            raise RuntimeError("门控模型未加载")
        active = [n for n in neurons if n.enabled]
        if not active:
            return []
        prompt = _build_gateway_prompt(query, active)
        with self._lock:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()
                      if k in ("input_ids", "attention_mask")}
            with torch.no_grad():
                out = self.model.generate(
                    **inputs, max_new_tokens=8, do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id)
            resp = self.tokenizer.decode(
                out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

        nums = re.findall(r'\d+', resp)
        idx = int(nums[0]) - 1 if nums else 0
        idx = max(0, min(idx, len(active) - 1))
        results = []
        for i, n in enumerate(active):
            results.append((n, 1.0 if i == idx else 0.1 / (abs(i - idx) + 1)))
        results.sort(key=lambda x: -x[1])
        return results

    def get_vram_mb(self):
        if self.model is None: return 0.0
        try:
            return sum(p.nelement() * p.element_size() for p in self.model.parameters()) / 1024**2
        except: return 0.0


# ═══════════════════════════════════════════════════
# 蜂群模式: 专家池 (按需加载/卸载)
# ═══════════════════════════════════════════════════

class ExpertPool:
    """GPU 上永远只有一个专家模型, 其余在 CPU 或磁盘."""

    def __init__(self, cache_in_cpu=True, max_cpu_cache=1):
        self.cache_in_cpu = cache_in_cpu
        self.max_cpu_cache = max_cpu_cache
        self._gpu_model = self._gpu_tok = self._gpu_name = None
        self._cpu_cache = {}    # {name: (model, tok)}
        self._cpu_order = []    # LRU
        self._lock = threading.Lock()

    def _evict_cpu(self):
        while len(self._cpu_cache) > self.max_cpu_cache:
            old = self._cpu_order.pop(0)
            if old in self._cpu_cache:
                del self._cpu_cache[old]; gc.collect()
                log(f"  🗑️ CPU 淘汰: {old}")

    def _to_cpu(self, name, model, tok):
        if not self.cache_in_cpu:
            del model; gc.collect(); return
        log(f"  💤 {name} → CPU")
        try:
            model.to("cpu")
            import torch; torch.cuda.empty_cache()
            self._cpu_cache[name] = (model, tok)
            if name in self._cpu_order: self._cpu_order.remove(name)
            self._cpu_order.append(name)
            self._evict_cpu()
        except Exception:
            del model; gc.collect()

    def _load_disk(self, neuron: NeuronDef):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        log(f"  📂 加载: {neuron.name} ({neuron.model_path})")
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(neuron.model_path, trust_remote_code=True)
        kw = {"device_map": "auto", "trust_remote_code": True}
        if neuron.quantize_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        else:
            kw["torch_dtype"] = torch.float16
        model = AutoModelForCausalLM.from_pretrained(neuron.model_path, **kw)
        log(f"  ✅ {neuron.name} ({time.time()-t0:.1f}s)")
        return model, tok

    def activate(self, neuron: NeuronDef):
        """确保专家在 GPU 上, 返回 (model, tok)"""
        import torch
        with self._lock:
            if self._gpu_name == neuron.name and self._gpu_model is not None:
                return self._gpu_model, self._gpu_tok

            if self._gpu_model is not None:
                log(f"  🔄 换出: {self._gpu_name}")
                self._to_cpu(self._gpu_name, self._gpu_model, self._gpu_tok)
                self._gpu_model = self._gpu_tok = self._gpu_name = None
                gc.collect()
                try: torch.cuda.empty_cache()
                except: pass

            if neuron.name in self._cpu_cache:
                log(f"  🔥 {neuron.name}: CPU → GPU")
                t0 = time.time()
                model, tok = self._cpu_cache.pop(neuron.name)
                if neuron.name in self._cpu_order: self._cpu_order.remove(neuron.name)
                try:
                    model.to("cuda")
                    self._gpu_model, self._gpu_tok, self._gpu_name = model, tok, neuron.name
                    log(f"  ✅ CPU→GPU ({time.time()-t0:.1f}s)")
                    return model, tok
                except:
                    del model; gc.collect()

            model, tok = self._load_disk(neuron)
            self._gpu_model, self._gpu_tok, self._gpu_name = model, tok, neuron.name
            return model, tok

    def unload_all(self):
        self._gpu_model = self._gpu_tok = self._gpu_name = None
        self._cpu_cache.clear(); self._cpu_order.clear()
        gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except: pass

    def status(self):
        size = 0.0
        if self._gpu_model:
            try: size = sum(p.nelement() * p.element_size() for p in self._gpu_model.parameters()) / 1024**2
            except: pass
        return {"gpu": self._gpu_name, "cpu": list(self._cpu_cache.keys()), "vram_mb": size}


# ═══════════════════════════════════════════════════
# 蜂群模式: NeuronSwarm 总控
# ═══════════════════════════════════════════════════

class NeuronSwarm:
    """小门控 + 多专家模型, 显存只占一个."""

    def __init__(self):
        self.config: Optional[NeuronConfig] = None
        self.gateway = GatewayRouter()
        self.pool = ExpertPool()
        self._started = False

    def init(self, config: NeuronConfig):
        self.config = config
        self.pool = ExpertPool(config.cache_in_cpu, config.max_cpu_cache)

    def start(self):
        if not self.config: raise RuntimeError("未初始化")
        if self.config.route_mode in ("gateway_llm", "hybrid"):
            if not self.config.gateway_model:
                raise RuntimeError("需要指定 gateway_model")
            self.gateway.load(self.config.gateway_model, self.config.gateway_quantize)
        self._started = True
        log(f"🐝 Swarm 启动: {len(self.config.neurons)} 专家 | "
            f"门控={self.config.route_mode}")

    def shutdown(self):
        self.gateway.unload(); self.pool.unload_all()
        self._started = False

    def route(self, query: str):
        if not self.config: return []
        neurons = self.config.neurons
        mode = self.config.route_mode
        if mode == "keyword":
            return keyword_route(query, neurons)
        elif mode == "gateway_llm":
            return self.gateway.route(query, neurons)
        elif mode == "hybrid":
            kw = keyword_route(query, neurons)
            if len(kw) >= 2 and kw[0][1] > kw[1][1] * 3:
                return kw
            return self.gateway.route(query, neurons)
        return keyword_route(query, neurons)

    def query(self, query: str, chat_history=None, max_new_tokens=512,
              temperature=0.7, stream=False):
        import torch
        if not self._started: raise RuntimeError("请先 start()")
        t0 = time.time()

        rt0 = time.time()
        scores = self.route(query)
        if not scores: return {"error": "无可用专家", "answer": ""}
        route_ms = (time.time() - rt0) * 1000
        chosen, chosen_score = scores[0]

        lt0 = time.time()
        model, tok = self.pool.activate(chosen)
        load_ms = (time.time() - lt0) * 1000

        gt0 = time.time()
        messages = []
        if chosen.description:
            messages.append({"role": "system", "content": f"你是{chosen.name}，{chosen.description}。"})
        if chat_history: messages.extend(chat_history)
        messages.append({"role": "user", "content": query})

        try:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            text = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"

        inputs = tok(text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
        gen_kw = {"max_new_tokens": max_new_tokens, "temperature": max(temperature, 0.01),
                  "do_sample": temperature > 0.01, "pad_token_id": tok.eos_token_id}

        if stream:
            return self._stream(model, tok, inputs, gen_kw, chosen, scores, route_ms, load_ms, t0)

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kw)
        answer = tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        gen_ms = (time.time() - gt0) * 1000
        total_ms = (time.time() - t0) * 1000
        chosen.total_calls += 1
        n = chosen.total_calls
        chosen.avg_latency_ms = (chosen.avg_latency_ms * (n-1) + total_ms) / n

        return {
            "answer": answer, "expert": chosen.name, "domain": chosen.domain,
            "route_scores": [(n.name, s) for n, s in scores[:5]],
            "timing": {"route_ms": round(route_ms), "load_ms": round(load_ms),
                       "gen_ms": round(gen_ms), "total_ms": round(total_ms)},
            "pool_status": self.pool.status(),
        }

    def _stream(self, model, tok, inputs, gen_kw, chosen, scores, route_ms, load_ms, t0):
        from transformers import TextIteratorStreamer
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        gen_kw["streamer"] = streamer
        thread = threading.Thread(target=lambda: model.generate(**inputs, **gen_kw))
        thread.start()
        full = ""
        for chunk in streamer:
            full += chunk
            yield {"chunk": chunk, "expert": chosen.name, "done": False}
        thread.join()
        yield {"chunk": "", "answer": full, "expert": chosen.name, "done": True,
               "timing": {"route_ms": round(route_ms), "load_ms": round(load_ms),
                           "total_ms": round((time.time()-t0)*1000)}}

    def get_status_md(self):
        if not self.config: return "⚠️ 未配置"
        st = self.pool.status()
        lines = [f"### 🐝 蜂群状态",
                 f"**门控**: {self.config.gateway_model or '关键词'} (~{self.gateway.get_vram_mb():.0f}MB)",
                 f"**GPU**: {st['gpu'] or '空'} ({st['vram_mb']:.0f}MB) | "
                 f"**CPU 缓存**: {', '.join(st['cpu']) or '无'}", ""]
        for n in self.config.neurons:
            s = "🟢" if n.enabled else "🔴"
            g = " ⚡GPU" if st["gpu"] == n.name else ""
            c = " 💤CPU" if n.name in st["cpu"] else ""
            lines.append(f"{s} **{n.name}** [{n.domain}] `{n.model_path}`{g}{c}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════
# 显存估算 (蜂群模式)
# ═══════════════════════════════════════════════════

def estimate_vram(config: NeuronConfig) -> str:
    lines = ["### 📊 显存估算\n"]
    gw = config.gateway_model
    gw_s = ("~300MB" if "0.5B" in gw else "~700MB" if "1.5B" in gw
            else "~1.5GB" if "3B" in gw else "~500MB") if gw else "0 (关键词模式)"
    lines.append(f"🚪 门控 (`{gw or '无'}`): **{gw_s}** {'(常驻)' if gw else ''}")

    if config.neurons:
        lines.append(f"\n专家 (一次只加载一个):")
        for n in config.neurons:
            p = n.model_path
            for pat, sz in [("0.5B", "~0.5GB"), ("1.5B", "~1GB"), ("1B", "~0.8GB"),
                            ("3B", "~2.5GB"), ("4B", "~3GB"), ("7B", "~5GB"),
                            ("8B", "~5.5GB"), ("9B", "~6GB"), ("14B", "~8GB"),
                            ("32B", "~18GB"), ("70B", "~36GB")]:
                if pat in p:
                    size = sz; break
            else:
                size = "~5GB"
            lines.append(f"  • {n.name}: `{p}` → **{size}** (4bit)" if n.quantize_4bit
                         else f"  • {n.name}: `{p}` → **{size}** (fp16, 约2×)")

    lines.append(f"\n**峰值** ≈ 门控 + 最大单个专家 | **CPU RAM** ≈ 缓存 {config.max_cpu_cache} 个")
    return "\n".join(lines)
