# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

# core/neuron_swarm.py - 神经元蜂群: 模型级专家路由
#
# 架构:
#   ┌─────────────┐
#   │ 门控小模型   │ ← 常驻 VRAM (~0.5GB)
#   │ (Qwen-0.5B) │
#   └──────┬──────┘
#          │ "这是医学问题 → 激活专家2"
#          ▼
#   ┌──────┐ ┌──────┐ ┌──────┐
#   │ 专家1 │ │ 专家2 │ │ 专家3 │  ← 磁盘/CPU，按需加载
#   │ 代码  │ │ 医学  │ │ 法律  │
#   └──────┘ └──────┘ └──────┘
#
# VRAM = 门控(~500MB) + 当前专家(~4-8GB 4bit)
# 总参数可以几百亿, 但显存永远只占一个模型的量
#
# 优势: 专家可以是完全不同的模型!
#   - 医学 → MedGemma / HuatuoGPT
#   - 代码 → DeepSeek-Coder / CodeQwen
#   - 通用 → Qwen / Yi / Mistral
#   - 甚至不同大小! 简单问题用3B, 复杂用9B

import gc
import json
import time
import torch
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from core.logger import log

SWARM_DIR = Path("data/neuron_swarm")
SWARM_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════

@dataclass
class ExpertNode:
    """一个专家节点 = 一个完整模型"""
    name: str                      # "医学专家"
    model_path: str                # HF ID 或本地路径
    domain: str = ""               # "医学"
    description: str = ""          # "擅长诊断、用药、病理分析"
    keywords: List[str] = field(default_factory=list)  # 辅助路由关键词
    priority: int = 5              # 0-10, 越高越优先
    quantize_4bit: bool = True     # 4bit 量化 (省显存)
    color: str = "#3B82F6"
    enabled: bool = True
    # 运行时统计
    total_calls: int = 0
    avg_latency_ms: float = 0.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass
class SwarmConfig:
    """蜂群配置"""
    name: str = "default"
    # 门控模型 (小模型, 常驻显存)
    gateway_model: str = ""        # HF ID, 如 "Qwen/Qwen2.5-0.5B-Instruct"
    gateway_quantize: bool = False # 门控一般不需要量化 (本身就小)
    # 专家列表
    experts: List[ExpertNode] = field(default_factory=list)
    # 路由模式
    route_mode: str = "gateway_llm"  # gateway_llm / keyword / hybrid
    # 专家缓存 (CPU RAM)
    cache_in_cpu: bool = True      # 卸载到 CPU 而非完全释放 (下次加载更快)
    max_cpu_cache: int = 1         # CPU 中最多缓存几个专家模型

    def to_dict(self):
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d):
        experts = [ExpertNode.from_dict(e) for e in d.pop("experts", [])]
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        cfg = cls(**{k: v for k, v in d.items() if k in valid})
        cfg.experts = experts
        return cfg


# ═══════════════════════════════════════════════════
# 门控路由器 (三种模式)
# ═══════════════════════════════════════════════════

# 预置领域关键词库
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


def _keyword_route(query: str, experts: List[ExpertNode]) -> List[Tuple[ExpertNode, float]]:
    """关键词路由 — 快速, 无需模型"""
    scores = []
    for expert in experts:
        if not expert.enabled:
            continue
        score = 0.0
        q = query.lower()
        # 自定义关键词
        for kw in expert.keywords:
            if kw.lower() in q:
                score += 3.0
        # 领域预置关键词
        for kw in DOMAIN_KEYWORDS.get(expert.domain, []):
            if kw.lower() in q:
                score += 1.0
        # 优先级加成
        score += expert.priority * 0.1
        # 兜底
        if score < 0.1:
            score = 0.1
        scores.append((expert, score))
    scores.sort(key=lambda x: -x[1])
    return scores


def _build_gateway_prompt(query: str, experts: List[ExpertNode]) -> str:
    """构建门控 LLM 的路由提示"""
    expert_desc = "\n".join(
        f"  {i+1}. [{e.name}] 领域: {e.domain} — {e.description or '通用'}"
        for i, e in enumerate(experts) if e.enabled
    )
    return (
        "你是一个智能路由器。根据用户的问题，选择最合适的专家来回答。\n"
        "只回复专家编号（数字），不要解释。\n\n"
        f"可用专家:\n{expert_desc}\n\n"
        f"用户问题: {query}\n\n"
        "最合适的专家编号是: "
    )


class GatewayRouter:
    """门控路由器 — 用小模型决定激活哪个专家"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_path = None
        self._lock = threading.Lock()

    def load(self, model_path: str, quantize: bool = False):
        """加载门控小模型"""
        if self.model_path == model_path and self.model is not None:
            return  # 已加载

        self.unload()
        log(f"🚪 加载门控模型: {model_path}")
        t0 = time.time()

        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        load_kw = {"device_map": "auto", "trust_remote_code": True}
        if quantize:
            load_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            load_kw["torch_dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kw)

        self.model = model
        self.tokenizer = tok
        self.model_path = model_path
        log(f"🚪 门控模型就绪 ({time.time()-t0:.1f}s)")

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
            self.tokenizer = None
            self.model_path = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def route(self, query: str, experts: List[ExpertNode]) -> List[Tuple[ExpertNode, float]]:
        """用门控 LLM 路由"""
        if self.model is None:
            raise RuntimeError("门控模型未加载")

        active = [e for e in experts if e.enabled]
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
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(
                out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
            ).strip()

        # 解析回复中的数字
        import re
        nums = re.findall(r'\d+', response)
        chosen_idx = int(nums[0]) - 1 if nums else 0
        chosen_idx = max(0, min(chosen_idx, len(active) - 1))

        # 构建得分列表 (选中的=1.0, 其余递减)
        results = []
        for i, e in enumerate(active):
            if i == chosen_idx:
                results.append((e, 1.0))
            else:
                results.append((e, 0.1 / (abs(i - chosen_idx) + 1)))
        results.sort(key=lambda x: -x[1])
        return results

    def get_vram_mb(self) -> float:
        """门控模型占用的显存"""
        if self.model is None:
            return 0.0
        try:
            return sum(
                p.nelement() * p.element_size()
                for p in self.model.parameters()
            ) / 1024 / 1024
        except Exception:
            return 0.0


# ═══════════════════════════════════════════════════
# 专家池 — 按需加载/卸载模型
# ═══════════════════════════════════════════════════

class ExpertPool:
    """
    管理多个专家模型 — 显存里永远只有一个.
    
    加载策略:
      - GPU: 当前活跃专家 (4bit 量化)
      - CPU: 最近使用的 N 个专家 (可选缓存)
      - 磁盘: 其余专家
    """

    def __init__(self, cache_in_cpu: bool = True, max_cpu_cache: int = 1):
        self.cache_in_cpu = cache_in_cpu
        self.max_cpu_cache = max_cpu_cache

        self._gpu_model = None       # 当前 GPU 上的模型
        self._gpu_name = None        # 当前 GPU 模型名
        self._gpu_tok = None         # 当前 tokenizer

        self._cpu_cache = {}         # {name: (model, tok)} CPU 缓存
        self._cpu_order = []         # LRU 顺序

        self._lock = threading.Lock()

    def _evict_cpu_cache(self):
        """LRU 淘汰 CPU 缓存"""
        while len(self._cpu_cache) > self.max_cpu_cache:
            oldest = self._cpu_order.pop(0)
            if oldest in self._cpu_cache:
                del self._cpu_cache[oldest]
                gc.collect()
                log(f"  🗑️ CPU 缓存淘汰: {oldest}")

    def _move_to_cpu(self, name: str, model, tok):
        """将模型从 GPU 移到 CPU"""
        if not self.cache_in_cpu:
            del model
            gc.collect()
            return

        log(f"  💤 {name} → CPU 缓存")
        try:
            model.to("cpu")
            torch.cuda.empty_cache()
            self._cpu_cache[name] = (model, tok)
            if name in self._cpu_order:
                self._cpu_order.remove(name)
            self._cpu_order.append(name)
            self._evict_cpu_cache()
        except Exception as e:
            log(f"  ⚠️ CPU 缓存失败: {e}, 直接释放")
            del model
            gc.collect()

    def _load_from_disk(self, expert: ExpertNode):
        """从磁盘加载模型到 GPU"""
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        log(f"  📂 加载专家: {expert.name} ({expert.model_path})")
        t0 = time.time()

        tok = AutoTokenizer.from_pretrained(
            expert.model_path, trust_remote_code=True)

        load_kw = {"device_map": "auto", "trust_remote_code": True}
        if expert.quantize_4bit:
            load_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            load_kw["torch_dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            expert.model_path, **load_kw)

        dt = time.time() - t0
        log(f"  ✅ {expert.name} 加载完成 ({dt:.1f}s)")
        return model, tok

    def activate(self, expert: ExpertNode) -> Tuple[Any, Any]:
        """
        激活一个专家 → 确保它在 GPU 上.
        
        流程:
          1. 已在 GPU? → 直接返回
          2. 在 CPU 缓存? → 移到 GPU (快)
          3. 都没有? → 从磁盘加载 (慢)
        """
        with self._lock:
            # 已在 GPU
            if self._gpu_name == expert.name and self._gpu_model is not None:
                log(f"  ⚡ {expert.name} 已在 GPU")
                return self._gpu_model, self._gpu_tok

            # 先卸载当前 GPU 模型
            if self._gpu_model is not None:
                log(f"  🔄 换出: {self._gpu_name}")
                self._move_to_cpu(self._gpu_name, self._gpu_model, self._gpu_tok)
                self._gpu_model = None
                self._gpu_name = None
                self._gpu_tok = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # 尝试从 CPU 缓存恢复
            if expert.name in self._cpu_cache:
                log(f"  🔥 {expert.name}: CPU → GPU (快速恢复)")
                t0 = time.time()
                model, tok = self._cpu_cache.pop(expert.name)
                if expert.name in self._cpu_order:
                    self._cpu_order.remove(expert.name)
                try:
                    model.to("cuda")
                    self._gpu_model = model
                    self._gpu_tok = tok
                    self._gpu_name = expert.name
                    log(f"  ✅ CPU→GPU 恢复 ({time.time()-t0:.1f}s)")
                    return model, tok
                except Exception as e:
                    log(f"  ⚠️ CPU→GPU 失败: {e}, 重新从磁盘加载")
                    del model
                    gc.collect()

            # 从磁盘加载
            model, tok = self._load_from_disk(expert)
            self._gpu_model = model
            self._gpu_tok = tok
            self._gpu_name = expert.name
            return model, tok

    def unload_all(self):
        """释放所有模型"""
        self._gpu_model = None
        self._gpu_name = None
        self._gpu_tok = None
        self._cpu_cache.clear()
        self._cpu_order.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log("🧹 专家池已清空")

    def status(self) -> Dict:
        """当前状态"""
        return {
            "gpu_expert": self._gpu_name,
            "cpu_cached": list(self._cpu_cache.keys()),
            "gpu_vram_mb": self._get_gpu_model_size(),
        }

    def _get_gpu_model_size(self) -> float:
        if self._gpu_model is None:
            return 0.0
        try:
            return sum(
                p.nelement() * p.element_size()
                for p in self._gpu_model.parameters()
            ) / 1024 / 1024
        except Exception:
            return 0.0


# ═══════════════════════════════════════════════════
# NeuronSwarm — 总控
# ═══════════════════════════════════════════════════

class NeuronSwarm:
    """
    神经元蜂群 — 小门控 + 多专家模型.
    
    用法:
      swarm = NeuronSwarm()
      swarm.load_config("my_swarm")  # 或 swarm.init(config)
      swarm.start()                  # 加载门控
      answer = swarm.query("患者头痛怎么办?")
      swarm.shutdown()
    """

    def __init__(self):
        self.config: Optional[SwarmConfig] = None
        self.gateway = GatewayRouter()
        self.pool = ExpertPool()
        self._started = False

    def init(self, config: SwarmConfig):
        """初始化配置"""
        self.config = config
        self.pool = ExpertPool(
            cache_in_cpu=config.cache_in_cpu,
            max_cpu_cache=config.max_cpu_cache,
        )

    def start(self):
        """启动: 加载门控模型"""
        if not self.config:
            raise RuntimeError("未初始化配置")
        if self.config.route_mode in ("gateway_llm", "hybrid"):
            if not self.config.gateway_model:
                raise RuntimeError("门控 LLM 模式需要指定 gateway_model")
            self.gateway.load(
                self.config.gateway_model,
                self.config.gateway_quantize,
            )
        self._started = True
        log(f"🐝 NeuronSwarm 启动: {len(self.config.experts)} 个专家 | "
            f"门控={self.config.route_mode}")

    def shutdown(self):
        """关闭: 释放所有模型"""
        self.gateway.unload()
        self.pool.unload_all()
        self._started = False
        log("🐝 NeuronSwarm 已关闭")

    def route(self, query: str) -> List[Tuple[ExpertNode, float]]:
        """路由: 返回 (专家, 得分) 列表"""
        if not self.config:
            return []
        experts = self.config.experts

        mode = self.config.route_mode
        if mode == "keyword":
            return _keyword_route(query, experts)
        elif mode == "gateway_llm":
            return self.gateway.route(query, experts)
        elif mode == "hybrid":
            # 先关键词粗筛, 再门控精排
            kw_scores = _keyword_route(query, experts)
            # 如果关键词有明确胜出 (>3x 第二名), 直接用
            if len(kw_scores) >= 2 and kw_scores[0][1] > kw_scores[1][1] * 3:
                return kw_scores
            # 否则用门控
            return self.gateway.route(query, experts)
        return _keyword_route(query, experts)

    def query(
        self,
        query: str,
        chat_history: Optional[List[Dict]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Dict:
        """
        完整查询流程:
          1. 路由 → 选专家
          2. 加载专家 (如需)
          3. 生成回答
          4. 返回结果 + 元信息
        """
        if not self._started:
            raise RuntimeError("请先调用 start()")

        t0 = time.time()

        # Step 1: 路由
        route_t0 = time.time()
        scores = self.route(query)
        if not scores:
            return {"error": "无可用专家", "answer": ""}
        route_ms = (time.time() - route_t0) * 1000

        chosen_expert, chosen_score = scores[0]
        log(f"🎯 路由选择: {chosen_expert.name} (domain={chosen_expert.domain}, "
            f"score={chosen_score:.2f}, {route_ms:.0f}ms)")

        # Step 2: 加载专家
        load_t0 = time.time()
        model, tok = self.pool.activate(chosen_expert)
        load_ms = (time.time() - load_t0) * 1000

        # Step 3: 生成
        gen_t0 = time.time()

        # 构建消息
        messages = []
        if chosen_expert.description:
            messages.append({
                "role": "system",
                "content": f"你是{chosen_expert.name}，{chosen_expert.description}。"
            })
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": query})

        # 尝试 chat template
        try:
            input_text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # fallback
            input_text = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"

        inputs = tok(input_text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()
                  if k in ("input_ids", "attention_mask")}

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": max(temperature, 0.01),
            "do_sample": temperature > 0.01,
            "pad_token_id": tok.eos_token_id,
        }

        if stream:
            return self._stream_generate(
                model, tok, inputs, gen_kwargs,
                chosen_expert, scores, route_ms, load_ms, t0)

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)

        answer = tok.decode(
            out[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        ).strip()
        gen_ms = (time.time() - gen_t0) * 1000

        # 更新统计
        total_ms = (time.time() - t0) * 1000
        chosen_expert.total_calls += 1
        n = chosen_expert.total_calls
        chosen_expert.avg_latency_ms = (
            chosen_expert.avg_latency_ms * (n - 1) + total_ms) / n

        return {
            "answer": answer,
            "expert": chosen_expert.name,
            "domain": chosen_expert.domain,
            "route_scores": [(e.name, s) for e, s in scores[:5]],
            "timing": {
                "route_ms": round(route_ms, 1),
                "load_ms": round(load_ms, 1),
                "gen_ms": round(gen_ms, 1),
                "total_ms": round(total_ms, 1),
            },
            "pool_status": self.pool.status(),
        }

    def _stream_generate(self, model, tok, inputs, gen_kwargs,
                         expert, scores, route_ms, load_ms, t0):
        """流式生成 (yield tokens)"""
        from transformers import TextIteratorStreamer
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer

        thread = threading.Thread(
            target=lambda: model.generate(**inputs, **gen_kwargs))
        thread.start()

        full_text = ""
        for chunk in streamer:
            full_text += chunk
            yield {
                "chunk": chunk,
                "expert": expert.name,
                "domain": expert.domain,
                "done": False,
            }

        thread.join()
        total_ms = (time.time() - t0) * 1000
        expert.total_calls += 1

        yield {
            "chunk": "",
            "answer": full_text,
            "expert": expert.name,
            "domain": expert.domain,
            "route_scores": [(e.name, s) for e, s in scores[:5]],
            "timing": {"route_ms": round(route_ms), "load_ms": round(load_ms),
                       "total_ms": round(total_ms)},
            "done": True,
        }

    def get_status_markdown(self) -> str:
        """UI 用的状态显示"""
        if not self.config:
            return "⚠️ 未加载配置"

        lines = [f"## 🐝 NeuronSwarm: {self.config.name}"]
        lines.append(f"**门控**: {self.config.gateway_model or '关键词模式'} "
                     f"(~{self.gateway.get_vram_mb():.0f}MB)")

        # 专家池状态
        st = self.pool.status()
        lines.append(f"**GPU 活跃**: {st['gpu_expert'] or '无'} "
                     f"({st['gpu_vram_mb']:.0f}MB)")
        lines.append(f"**CPU 缓存**: {', '.join(st['cpu_cached']) or '无'}")
        lines.append("")

        # 专家列表
        for i, e in enumerate(self.config.experts):
            status = "🟢" if e.enabled else "🔴"
            gpu_mark = " ⚡GPU" if st['gpu_expert'] == e.name else ""
            cpu_mark = " 💤CPU" if e.name in st['cpu_cached'] else ""
            lines.append(
                f"{status} **{e.name}** [{e.domain}] "
                f"— {e.model_path}{gpu_mark}{cpu_mark}\n"
                f"   调用 {e.total_calls} 次 | "
                f"平均 {e.avg_latency_ms:.0f}ms | "
                f"4bit={'✅' if e.quantize_4bit else '❌'}"
            )
        return "\n".join(lines)

    def preview_route(self, query: str) -> str:
        """路由预览 (纯文本, 不加载模型)"""
        if not self.config:
            return "⚠️ 未加载配置"

        # 只用关键词路由做预览 (不需要门控模型)
        scores = _keyword_route(query, self.config.experts)
        if not scores:
            return "❌ 无可用专家"

        max_s = max(s for _, s in scores) if scores else 1
        lines = [f"### 🔍 路由预览 (关键词模式)\n**查询**: {query}\n"]
        for i, (expert, score) in enumerate(scores[:5]):
            medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
            pct = score / max_s * 100 if max_s > 0 else 0
            bar = "█" * int(pct / 2.5) + "░" * (40 - int(pct / 2.5))
            hits = [kw for kw in expert.keywords if kw.lower() in query.lower()]
            domain_hits = [kw for kw in DOMAIN_KEYWORDS.get(expert.domain, [])
                          if kw.lower() in query.lower()]
            lines.append(
                f"{medal} **{expert.name}** [{expert.domain}] — "
                f"得分: **{score:.1f}**\n"
                f"  `{bar}`\n"
                f"  模型: `{expert.model_path}`"
            )
            if hits or domain_hits:
                all_hits = hits + domain_hits
                lines.append(f"  命中: {', '.join(all_hits[:8])}")
            lines.append("")

        lines.append(f"---\n**将激活**: 🧠 {scores[0][0].name} "
                     f"(模型: `{scores[0][0].model_path}`)")
        if self.config.route_mode in ("gateway_llm", "hybrid"):
            lines.append(f"\n> 💡 实际推理时门控 LLM 会做最终决策 "
                         f"(当前预览仅用关键词)")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════
# 配置持久化
# ═══════════════════════════════════════════════════

def save_swarm_config(config: SwarmConfig):
    path = SWARM_DIR / f"{config.name}.json"
    path.write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log(f"💾 蜂群配置已保存: {path}")
    return str(path)


def load_swarm_config(name: str) -> SwarmConfig:
    path = SWARM_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"配置不存在: {path}")
    d = json.loads(path.read_text(encoding="utf-8"))
    return SwarmConfig.from_dict(d)


def list_swarm_configs() -> List[str]:
    return [p.stem for p in SWARM_DIR.glob("*.json")]


def estimate_vram(config: SwarmConfig) -> str:
    """估算显存需求"""
    lines = ["### 📊 显存估算\n"]

    # 门控
    gw = config.gateway_model
    gw_size = "~300MB" if "0.5B" in gw else "~700MB" if "1.5B" in gw else "~1.5GB" if "3B" in gw else "~500MB"
    lines.append(f"🚪 门控 ({gw or '关键词'}): **{gw_size}** (常驻)")

    # 最大专家
    if config.experts:
        lines.append(f"\n专家模型 (一次只加载一个):")
        for e in config.experts:
            p = e.model_path
            if "0.5B" in p or "0.5b" in p:
                size = "~0.5GB (4bit)"
            elif "1.5B" in p or "1.5b" in p or "1B" in p:
                size = "~1GB (4bit)"
            elif "3B" in p or "4B" in p:
                size = "~2.5GB (4bit)"
            elif "7B" in p or "8B" in p or "9B" in p:
                size = "~5GB (4bit)"
            elif "14B" in p or "13B" in p:
                size = "~8GB (4bit)"
            elif "32B" in p or "34B" in p:
                size = "~18GB (4bit)"
            elif "70B" in p or "72B" in p:
                size = "~36GB (4bit)"
            else:
                size = "~5GB (4bit)"
            q = "4bit" if e.quantize_4bit else "fp16"
            lines.append(f"  • {e.name}: `{p}` → **{size}**")

    lines.append(f"\n**峰值 VRAM** ≈ 门控 + 最大单个专家")
    lines.append(f"**CPU RAM** ≈ 缓存 {config.max_cpu_cache} 个专家 (可选)")
    return "\n".join(lines)
