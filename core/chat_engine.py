# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

# core/chat_engine.py - ForgeX v2.7 推理引擎（Chat 測試）
# 修复: Gradio 4.x messages 格式、流式输出、LoRA 自动检测、更好的错误处理
from pathlib import Path
from typing import List, Dict, Optional, Generator
import time
import traceback

from core import log


class ChatEngine:
    """多后端推理引擎 (HF / GGUF / API / 自动检测)"""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._pipe = None
        self._gguf = None
        self._backend: str = ""
        self._current_model_path: str = ""
        self._lora_path: Optional[str] = None
        self._model_name: str = ""
        self._load_error: str = ""

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None or self._gguf is not None or self._model is not None

    @property
    def model_name(self) -> str:
        return self._model_name or self._current_model_path or ""

    def load_model(
        self, model_path: str,
        lora_path: Optional[str] = None,
        backend: str = "auto",
    ) -> str:
        """载入模型

        backend: "auto" | "hf" | "gguf"
        """
        if not model_path or not model_path.strip():
            return "❌ 请输入模型路径"

        model_path = model_path.strip()
        backend = (backend or "auto").lower().strip()

        # 自动检测后端
        if backend == "auto":
            mp = Path(model_path)
            if mp.suffix.lower() == ".gguf":
                backend = "gguf"
            elif mp.is_file() and mp.stat().st_size > 100_000_000:
                backend = "gguf"
            else:
                backend = "hf"
            log(f"自动检测后端: {backend}")

        # 如果是同一个模型，跳过
        if (model_path == self._current_model_path
            and lora_path == self._lora_path
            and backend == self._backend
            and self.is_loaded):
            return f"✅ 模型已载入: {self._model_name}"

        self.unload()
        self._model_name = Path(model_path).name
        log(f"载入推理模型[{backend}]: {model_path}" +
            (f" + LoRA: {lora_path}" if (lora_path and backend == 'hf') else ""))

        try:
            if backend == "gguf":
                return self._load_gguf(model_path)
            else:
                return self._load_hf(model_path, lora_path)
        except Exception as e:
            self._load_error = str(e)
            log(f"载入失败: {e}\n{traceback.format_exc()}")
            return f"❌ 载入失败: {e}"

    def _load_gguf(self, model_path: str) -> str:
        try:
            from llama_cpp import Llama
        except ImportError:
            return (
                "❌ 未安装 llama-cpp-python。\n"
                "安装: pip install llama-cpp-python\n"
                "（GPU 版本: CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python）"
            )

        mp = Path(model_path)
        if not mp.exists():
            return f"❌ GGUF 文件不存在: {model_path}"

        n_gpu = -1  # 全部放 GPU
        try:
            self._gguf = Llama(
                model_path=str(mp),
                n_ctx=4096,
                n_gpu_layers=n_gpu,
                logits_all=False,
                verbose=False,
            )
            self._backend = "gguf"
            self._current_model_path = str(mp)
            self._lora_path = None
            return f"✅ GGUF 已载入: {mp.name}"
        except Exception as e:
            self.unload()
            return f"❌ GGUF 载入失败: {e}"

    def _load_hf(self, model_path: str, lora_path: Optional[str] = None) -> str:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import torch
        except ImportError:
            return "❌ 未安装 transformers。请执行: pip install transformers torch"

        mp = Path(model_path)
        has_adapter_cfg = mp.is_dir() and (mp / "adapter_config.json").exists()
        has_model_cfg = mp.is_dir() and (mp / "config.json").exists()
        has_safetensors = mp.is_dir() and bool(list(mp.glob("*.safetensors")) + list(mp.glob("*.bin")))
        is_hf_id = not mp.exists() and "/" in model_path  # HuggingFace Hub ID

        # 判断是 LoRA 还是完整模型:
        # - 纯 LoRA: 有 adapter_config.json, 没有 config.json (或没有模型权重文件)
        # - 完整模型: 有 config.json + 权重文件
        # - 合并残留: 有 adapter_config + config.json + 权重 → 当作完整模型
        is_lora_dir = has_adapter_cfg and not (has_model_cfg and has_safetensors)
        is_full_model = has_model_cfg and has_safetensors

        # 如果用户指定了一个 LoRA 目录作为模型路径，自动读取基座
        base_model = model_path
        actual_lora = lora_path
        if is_lora_dir and not is_full_model:
            try:
                import json
                acfg = json.loads((mp / "adapter_config.json").read_text(encoding="utf-8"))
                raw_base = acfg.get("base_model_name_or_path", "")
                actual_lora = str(mp)
                if not raw_base:
                    return "❌ LoRA 的 adapter_config.json 中未记录基座模型路径，请在「LoRA路径」栏另外填基座模型路径"
                # ═══ 修复过时路径 ═══
                from core.distiller import _repair_base_model_path
                resolved = _repair_base_model_path(raw_base)
                if resolved:
                    base_model = resolved
                    if resolved != raw_base:
                        log(f"🔧 基座路径修复: {Path(raw_base).name} → {Path(resolved).name}")
                else:
                    base_model = raw_base  # 让后续加载尝试（可能是 HF hub ID）
                log(f"自动检测 LoRA → 基座: {base_model}, LoRA: {actual_lora}")
            except ImportError:
                # _repair_base_model_path 不可用时退回
                base_model = acfg.get("base_model_name_or_path", model_path)
                log(f"自动检测 LoRA → 基座: {base_model}")
            except Exception as e:
                return f"❌ 读取 adapter_config.json 失败: {e}"

        # 检查路径是否有效
        base_path = Path(base_model)
        if base_path.exists() and not (base_path / "config.json").exists() and not is_hf_id:
            return f"❌ 目录 {base_model} 中没有 config.json，不是有效的 HF 模型目录"

        try:
            dtype = torch.float16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                dtype = torch.bfloat16

            # 预检查: trust_remote_code 模型是否缺少自定义 .py 文件
            try:
                self._precheck_custom_code(base_model)
            except Exception as pre_err:
                log(f"⚠️ 预检查自定义代码时出错（不影响加载）: {pre_err}")

            log(f"加载 tokenizer: {base_model}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                base_model, trust_remote_code=True,
            )

            log(f"加载模型: {base_model} (dtype={dtype})")
            # 预导入模型类（防止 LazyAutoMapping 失效）
            try:
                from core.safe_loader import ensure_model_importable, dtype_kwarg
                ensure_model_importable(base_model)
            except Exception:
                pass

            # 构建模型加载参数（兼容新旧 transformers 的 dtype 参数名）
            try:
                _mk = {**dtype_kwarg(dtype), "device_map": "auto",
                       "trust_remote_code": True, "low_cpu_mem_usage": True}
            except Exception:
                _mk = {"torch_dtype": dtype, "device_map": "auto",
                       "trust_remote_code": True, "low_cpu_mem_usage": True}

            try:
                self._model = AutoModelForCausalLM.from_pretrained(base_model, **_mk)
            except Exception as load_err:
                err_msg = str(load_err)

                # "Could not find XxxForCausalLM" → safe_load fallback
                if "Could not find" in err_msg:
                    log(f"⚠️ auto-mapping 失败，使用 safe_loader 重试...")
                    from core.safe_loader import safe_load_model
                    self._model = safe_load_model(base_model, **_mk)
                # trust_remote_code 模型缺少自定义 .py 文件
                # transformers 不同版本可能抛 OSError / ValueError / ImportError
                elif "does not appear to have a file named" in err_msg and ".py" in err_msg:
                    log("⚠️ 检测到缺少自定义模型代码文件，尝试自动修复...")
                    fixed = self._try_fix_custom_code(base_model)
                    if fixed:
                        log("🔧 已复制自定义代码，重试加载...")
                        self._model = AutoModelForCausalLM.from_pretrained(base_model, **_mk)
                    else:
                        raise
                else:
                    raise

            # 挂载 LoRA
            if actual_lora:
                try:
                    from peft import PeftModel
                    log(f"挂载 LoRA: {actual_lora}")
                    self._model = PeftModel.from_pretrained(self._model, actual_lora)
                    self._model = self._model.merge_and_unload()
                    log(f"✅ LoRA 已合并: {actual_lora}")
                except ImportError:
                    log("⚠️ 未安装 peft，跳过 LoRA 挂载")
                except Exception as e:
                    log(f"⚠️ LoRA 挂载失败（将使用基座模型）: {e}")

            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._pipe = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
            )
            self._backend = "hf"
            self._current_model_path = model_path
            self._lora_path = actual_lora or lora_path
            name = Path(base_model).name
            lora_note = f" + LoRA({Path(actual_lora).name})" if actual_lora else ""

            # 显示显存占用
            vram_info = ""
            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated() / 1024**3
                total = torch.cuda.get_device_properties(0).total_mem / 1024**3
                vram_info = f" | VRAM: {used:.1f}/{total:.1f} GB"

            return f"✅ 已载入: {name}{lora_note}{vram_info}"

        except Exception as e:
            self.unload()
            err = str(e)
            if "CUDA out of memory" in err or "OutOfMemoryError" in err:
                return f"❌ 显存不足。建议: 1) 使用更小的模型 2) 关闭其他 GPU 进程 3) 使用 GGUF 格式\n详细: {err[:200]}"
            return f"❌ HF 模型载入失败: {err[:300]}"

    def _precheck_custom_code(self, model_path: str):
        """在加载前预检查并修复 trust_remote_code 模型缺少的自定义 .py 文件。"""
        import json as _json
        mp = Path(model_path)
        if not mp.is_dir():
            return
        config_path = mp / "config.json"
        if not config_path.exists():
            return
        try:
            cfg = _json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return
        auto_map = cfg.get("auto_map", {})
        if not auto_map:
            return
        # 检查是否缺少任何 .py 文件
        for val in auto_map.values():
            if isinstance(val, str):
                if "--" in val:
                    val = val.split("--", 1)[1]
                mod = val.split(".")[0]
                if mod and not (mp / f"{mod}.py").exists():
                    log(f"预检查: 缺少 {mod}.py，尝试自动修复...")
                    self._try_fix_custom_code(model_path)
                    return

    def _try_fix_custom_code(self, model_path: str) -> bool:
        """尝试修复 trust_remote_code 模型缺少的自定义 .py 文件。

        自包含实现，不强依赖 core.merger（避免 yaml 等间接依赖导致失败）。
        """
        import json as _json, shutil, os
        mp = Path(model_path)
        if not mp.is_dir():
            return False

        config_path = mp / "config.json"
        if not config_path.exists():
            return False

        try:
            cfg = _json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return False

        auto_map = cfg.get("auto_map", {})
        if not auto_map:
            return False

        # 收集缺失的 .py 文件
        needed = set()
        for val in auto_map.values():
            if isinstance(val, str):
                if "--" in val:
                    val = val.split("--", 1)[1]
                mod = val.split(".")[0]
                if mod:
                    needed.add(f"{mod}.py")
        missing = {f for f in needed if not (mp / f).exists()}
        if not missing:
            return True  # 已全部存在

        log(f"🔧 缺少自定义代码文件: {missing}")

        # 确定源模型
        source_model = None
        recipe_path = mp / "forgex_merge_recipe.json"
        if recipe_path.exists():
            try:
                recipe = _json.loads(recipe_path.read_text(encoding="utf-8"))
                source_model = recipe.get("base_model", "")
            except Exception:
                pass
        if not source_model:
            source_model = cfg.get("_name_or_path", "")
        if not source_model:
            log("⚠️ 无法确定源模型路径")
            return False

        # 找到源模型目录
        src_dir = None

        # 方法1: 优先用 merger 的 _resolve_hf_model_dir（最完整）
        try:
            from core.merger import _resolve_hf_model_dir
            src_dir = _resolve_hf_model_dir(source_model)
        except Exception:
            pass

        # 方法2: 直接检查本地路径
        if not src_dir:
            sp = Path(source_model)
            if sp.is_dir():
                src_dir = sp

        # 方法3: 手动扫描 HF cache
        if not src_dir:
            try:
                home = Path(os.path.expanduser("~"))
                safe_name = source_model.replace("/", "--")
                for cache_root in [
                    home / ".cache" / "huggingface" / "hub",
                ]:
                    model_dir = cache_root / f"models--{safe_name}"
                    snapshots = model_dir / "snapshots"
                    if snapshots.is_dir():
                        for snap in sorted(snapshots.iterdir(), reverse=True):
                            if snap.is_dir() and (snap / "config.json").exists():
                                src_dir = snap
                                break
                    if src_dir:
                        break
            except Exception:
                pass

        if not src_dir:
            log(f"⚠️ 无法找到源模型 {source_model} 的本地目录")
            return False

        # 复制所有自定义代码 .py 文件
        copied = []
        for py in src_dir.glob("*.py"):
            if py.name.startswith(("modeling_", "configuration_", "tokenization_", "image_processing_")):
                dst = mp / py.name
                if not dst.exists():
                    try:
                        shutil.copy2(py, dst)
                        copied.append(py.name)
                    except Exception as e:
                        log(f"⚠️ 复制 {py.name} 失败: {e}")

        # 修正 auto_map 中的 repo 前缀
        modified = False
        for key, val in auto_map.items():
            if isinstance(val, str) and "--" in val:
                auto_map[key] = val.split("--", 1)[1]
                modified = True
        if modified:
            cfg["auto_map"] = auto_map
            try:
                config_path.write_text(_json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

        if copied:
            log(f"✅ 已复制: {copied}")

        # 验证
        still_missing = {f for f in needed if not (mp / f).exists()}
        return len(still_missing) == 0

    def unload(self):
        """卸载模型并释放 VRAM"""
        for attr in ("_model", "_pipe", "_gguf"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    del obj
                except Exception:
                    pass
                setattr(self, attr, None)
        self._tokenizer = None
        self._backend = ""
        self._current_model_path = ""
        self._lora_path = None
        self._model_name = ""
        self._load_error = ""

        try:
            import gc; gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        log("模型已卸载")
        return "✅ 模型已卸载，VRAM 已释放"

    def chat(
        self,
        message: str,
        history: Optional[List[Dict]] = None,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
    ) -> str:
        """单轮/多轮对话"""
        if not self.is_loaded:
            return "❌ 请先载入模型（点击「📦 载入模型」按钮）"

        if not message or not message.strip():
            return ""

        # 构建消息列表
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})

        # 处理历史消息（兼容多种格式）
        if history:
            for h in history:
                if isinstance(h, dict):
                    role = h.get("role", "")
                    content = h.get("content", "")
                    if role and content:
                        messages.append({"role": role, "content": content})
                elif isinstance(h, (list, tuple)) and len(h) >= 2:
                    messages.append({"role": "user", "content": str(h[0])})
                    if h[1]:
                        messages.append({"role": "assistant", "content": str(h[1])})
        messages.append({"role": "user", "content": message.strip()})

        temperature = float(max(temperature, 0.01))
        max_new_tokens = int(max(max_new_tokens, 1))

        try:
            if self._backend == "gguf":
                return self._chat_gguf(messages, temperature, top_p, max_new_tokens)
            else:
                return self._chat_hf(messages, temperature, top_p, max_new_tokens)
        except Exception as e:
            log(f"推理异常: {e}\n{traceback.format_exc()}")
            return f"❌ 推理错误: {e}"

    def _chat_gguf(self, messages, temperature, top_p, max_new_tokens) -> str:
        try:
            out = self._gguf.create_chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=float(top_p),
            )
            txt = (out.get("choices") or [{}])[0].get("message", {}).get("content", "")
            return (txt or "").strip() or "(模型无输出)"
        except Exception as e1:
            log(f"create_chat_completion 失败 ({e1})，降级到原始 prompt")
            try:
                prompt = self._build_fallback_prompt(messages)
                out = self._gguf(
                    prompt,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=float(top_p),
                    stop=["</s>", "<|end|>", "<|im_end|>", "<|im_start|>",
                          "<|eot_id|>", "<|end_of_turn|>"],
                )
                txt = (out.get("choices") or [{}])[0].get("text", "")
                return (txt or "").strip() or "(模型无输出)"
            except Exception as e2:
                return f"❌ GGUF 推理错误: {e2}"

    def _chat_hf(self, messages, temperature, top_p, max_new_tokens) -> str:
        # 优先用 chat_template
        if self._tokenizer and hasattr(self._tokenizer, "apply_chat_template"):
            try:
                prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception as e:
                log(f"apply_chat_template 失败 ({e})，用 fallback")
                prompt = self._build_fallback_prompt(messages)
        else:
            prompt = self._build_fallback_prompt(messages)

        # 直接用 model.generate 替代 pipeline（更稳定）
        try:
            import torch
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            input_ids = inputs["input_ids"].to(self._model.device)
            attention_mask = inputs["attention_mask"].to(self._model.device)

            with torch.no_grad():
                output_ids = self._model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0.01 else None,
                    top_p=float(top_p) if temperature > 0.01 else None,
                    do_sample=temperature > 0.01,
                    pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
                )

            # 只取生成的部分
            new_tokens = output_ids[0][input_ids.shape[1]:]
            reply = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

            # 清理停止符（decode 后可能残留）
            for stop in ["<|im_end|>", "<|endoftext|>", "</s>", "<|end|>",
                         "<|eot_id|>", "<|end_of_turn|>", "<|assistant|>"]:
                if stop in reply:
                    reply = reply[:reply.index(stop)]

            return reply.strip() or "(模型无输出)"

        except Exception as e:
            log(f"model.generate 失败: {e}，尝试 pipeline")
            # Fallback: pipeline
            return self._chat_hf_pipeline(prompt, temperature, top_p, max_new_tokens)

    def _chat_hf_pipeline(self, prompt, temperature, top_p, max_new_tokens) -> str:
        """Pipeline fallback"""
        output = self._pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=float(top_p),
            do_sample=temperature > 0.01,
            pad_token_id=self._tokenizer.pad_token_id if self._tokenizer else None,
            return_full_text=True,
        )
        generated = output[0]["generated_text"]

        if isinstance(generated, str):
            reply = generated[len(prompt):] if generated.startswith(prompt) else generated
        elif isinstance(generated, list):
            reply = generated[-1].get("content", "") if generated else ""
        else:
            reply = str(generated)

        for stop in ["<|im_end|>", "<|endoftext|>", "</s>", "<|end|>",
                     "<|eot_id|>", "<|end_of_turn|>"]:
            if stop in reply:
                reply = reply[:reply.index(stop)]
        return reply.strip() or "(模型无输出)"

    def _build_fallback_prompt(self, messages) -> str:
        """构建通用 ChatML 格式 prompt"""
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    def get_status(self) -> Dict:
        status = {
            "loaded": self.is_loaded,
            "backend": self._backend,
            "model": self._current_model_path or "无",
            "lora": self._lora_path or "无",
        }
        if self.is_loaded:
            try:
                import torch
                if torch.cuda.is_available():
                    used = torch.cuda.memory_allocated() / 1024**3
                    total = torch.cuda.get_device_properties(0).total_mem / 1024**3
                    status["vram"] = f"{used:.1f}/{total:.1f} GB"
            except Exception:
                pass
        return status


chat_engine = ChatEngine()
