# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

# core/merger.py - ForgeX v2 MergeKit 合併器（修復 YAML 結構）
import json
import os
from pathlib import Path
from typing import List, Dict, Optional

try:
    import yaml
except ImportError:
    yaml = None  # mergekit 功能不可用，但不阻止启动

from core import LORAS_DIR, log, run_subprocess, get_timestamp, random_name, config
from core.task_queue import Task



def _peft_from_pretrained_with_offload(model, adapter_path: str, offload_dir: Path):
    """Load a PEFT adapter onto a potentially-dispatched base model.

    When the base model is loaded with device_map='auto', accelerate may require an offload folder
    during adapter weight loading / dispatch. Different PEFT versions accept different kwargs,
    so we try the richer signature first and fall back gracefully.
    """
    from peft import PeftModel
    has_map = bool(getattr(model, "hf_device_map", None))
    if not has_map:
        return PeftModel.from_pretrained(model, adapter_path)
    # Try common kwarg names across versions.
    for kwargs in (
        {"device_map": "auto", "offload_folder": str(offload_dir)},
        {"device_map": "auto", "offload_dir": str(offload_dir)},
        {"offload_folder": str(offload_dir)},
        {"offload_dir": str(offload_dir)},
        {},
    ):
        try:
            return PeftModel.from_pretrained(model, adapter_path, **kwargs)
        except TypeError:
            continue
    # Last resort without kwargs
    return PeftModel.from_pretrained(model, adapter_path)

def _read_adapter_config(adapter_dir: Path) -> Dict:
    """Read PEFT adapter_config.json if present."""
    try:
        import json
        p = adapter_dir / "adapter_config.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _resolve_hf_model_dir(model_name: str) -> Optional[Path]:
    """尝试找到 HuggingFace 模型的本地缓存目录。"""
    import os
    # 1. 如果已经是本地路径
    p = Path(model_name)
    if p.is_dir():
        return p

    # 2. snapshot_download (最可靠)
    try:
        from huggingface_hub import snapshot_download
        cache_dir = snapshot_download(model_name, local_files_only=True)
        return Path(cache_dir)
    except Exception:
        pass

    # 3. try_to_load_from_cache (轻量级)
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(model_name, "config.json")
        if cached and isinstance(cached, str) and Path(cached).exists():
            return Path(cached).parent
    except Exception:
        pass

    # 4. 手动扫描 HF cache 目录（兼容各版本/系统）
    cache_dirs = []
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        cache_dirs.append(Path(HF_HUB_CACHE))
    except Exception:
        pass
    home = Path(os.path.expanduser("~"))
    for c in [
        home / ".cache" / "huggingface" / "hub",
        Path(os.environ.get("HF_HOME", "")) / "hub" if os.environ.get("HF_HOME") else None,
        Path(os.environ.get("TRANSFORMERS_CACHE", "")) if os.environ.get("TRANSFORMERS_CACHE") else None,
    ]:
        if c and c.is_dir() and c not in cache_dirs:
            cache_dirs.append(c)

    safe_name = model_name.replace("/", "--")
    for cd in cache_dirs:
        model_dir = cd / f"models--{safe_name}"
        if model_dir.is_dir():
            snapshots = model_dir / "snapshots"
            if snapshots.is_dir():
                for snap in sorted(snapshots.iterdir(), reverse=True):
                    if snap.is_dir() and (snap / "config.json").exists():
                        return snap
            if (model_dir / "config.json").exists():
                return model_dir
    return None


def _copy_custom_model_code(source_model: str, output_dir: Path):
    """将 trust_remote_code 模型的自定义 .py 文件复制到输出目录。

    Phi-3、InternLM、ChatGLM 等模型在 config.json 中声明了 auto_map，
    指向自定义的 modeling_xxx.py / configuration_xxx.py / tokenization_xxx.py。
    save_pretrained() 只保存权重和 config，不会复制这些 .py 文件，
    导致从本地目录加载时报错：
    "does not appear to have a file named modeling_phi3.py"

    此函数检测 config.json 中的 auto_map，找到对应的 .py 文件，复制过去。
    如果本地缓存找不到，尝试从 HuggingFace 直接下载。
    """
    import shutil

    # 读取输出目录的 config.json 查看是否需要自定义代码
    config_path = output_dir / "config.json"
    if not config_path.exists():
        return

    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return

    auto_map = cfg.get("auto_map", {})
    if not auto_map:
        return

    # 收集需要的 .py 文件名
    needed_py_files = set()
    for key, value in auto_map.items():
        # value 格式: "modeling_phi3.Phi3ForCausalLM" 或 "microsoft/Phi-3--modeling_phi3.Phi3ForCausalLM"
        if isinstance(value, str):
            if "--" in value:
                value = value.split("--", 1)[1]
            module_name = value.split(".")[0]  # "modeling_phi3"
            if module_name:
                needed_py_files.add(f"{module_name}.py")

    if not needed_py_files:
        return

    # 检查哪些文件缺失
    missing_files = {f for f in needed_py_files if not (output_dir / f).exists()}
    if not missing_files:
        return

    log(f"🔧 检测到 trust_remote_code 模型，需要复制自定义代码: {missing_files}")

    def _safe_copy(src: Path, dst: Path):
        """安全复制（处理 Windows 只读文件等问题）"""
        try:
            shutil.copy2(src, dst)
            return True
        except PermissionError:
            try:
                # Windows: HF cache 文件可能只读，改用 read + write
                dst.write_bytes(src.read_bytes())
                return True
            except Exception as e2:
                log(f"⚠️ 复制失败 {src.name}: {e2}")
                return False
        except Exception as e:
            log(f"⚠️ 复制失败 {src.name}: {e}")
            return False

    # ---- 策略 1: 从本地缓存查找 ----
    src_dir = _resolve_hf_model_dir(source_model)
    copied = []
    if src_dir:
        for py_file in list(missing_files):
            src_file = src_dir / py_file
            if src_file.exists() and _safe_copy(src_file, output_dir / py_file):
                copied.append(py_file)
                missing_files.discard(py_file)

        # 额外：复制所有相关的 .py 文件（有些模型拆分成多个文件）
        for py in src_dir.glob("*.py"):
            if py.name.startswith(("modeling_", "configuration_", "tokenization_", "image_processing_")):
                dst = output_dir / py.name
                if not dst.exists():
                    if _safe_copy(py, dst):
                        copied.append(py.name)

    # ---- 策略 2: 从 HuggingFace 直接下载缺失文件 ----
    if missing_files and "/" in source_model and not Path(source_model).is_dir():
        log(f"📥 尝试从 HuggingFace 下载缺失文件: {missing_files}")
        try:
            from huggingface_hub import hf_hub_download
            for py_file in list(missing_files):
                try:
                    downloaded = hf_hub_download(
                        repo_id=source_model,
                        filename=py_file,
                    )
                    if downloaded and _safe_copy(Path(downloaded), output_dir / py_file):
                        copied.append(py_file)
                        missing_files.discard(py_file)
                except Exception as e:
                    log(f"  下载 {py_file} 失败: {e}")
        except ImportError:
            log("⚠️ huggingface_hub 未安装，无法下载缺失文件")
        except Exception as e:
            log(f"⚠️ HuggingFace 下载失败: {e}")

    # ---- 策略 3: 如果还有缺失，尝试用 _name_or_path 再搜索 ----
    if missing_files:
        name_or_path = cfg.get("_name_or_path", "")
        if name_or_path and name_or_path != source_model:
            alt_dir = _resolve_hf_model_dir(name_or_path)
            if alt_dir:
                for py_file in list(missing_files):
                    src_file = alt_dir / py_file
                    if src_file.exists() and _safe_copy(src_file, output_dir / py_file):
                        copied.append(py_file)
                        missing_files.discard(py_file)

    if copied:
        log(f"✅ 已复制自定义模型代码: {copied}")

    if missing_files:
        log(f"⚠️ 仍有文件缺失: {missing_files}")
        log(f"   请手动从 HuggingFace 下载 {source_model} 并复制以下文件到 {output_dir}:")
        for f in missing_files:
            log(f"   - {f}")

    # 修正 auto_map 中的 repo 前缀（从 "org/repo--module.Class" → "module.Class"）
    modified = False
    for key, value in auto_map.items():
        if isinstance(value, str) and "--" in value:
            auto_map[key] = value.split("--", 1)[1]
            modified = True
    if modified:
        cfg["auto_map"] = auto_map
        config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        log(f"✅ 已修正 config.json 中的 auto_map 路径前缀")


def _rewrite_adapter_keys(adapter_dir: Path, work_dir: Path) -> Path:
    """Create a rewritten adapter directory with normalized module key prefixes.

    Why:
      Some LoRA adapters were saved with extra wrapper hops in parameter keys, e.g.
        base_model.model.model.model.layers.10.input_layernorm
      while the runtime base model exposes only
        base_model.model.layers.10.input_layernorm

      A naive fix by setting `inner.model = inner` introduces a cyclic nn.Module tree
      (self-referential submodule) and can trigger `maximum recursion depth exceeded`.

    We instead rewrite adapter state-dict keys on disk and retry loading.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    # Copy all files first
    import shutil
    for item in adapter_dir.iterdir():
        dst = work_dir / item.name
        if item.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)

    def _normalize_key(k: str) -> str:
        # collapse repeated `.model.` hops after `base_model.`
        # e.g. base_model.model.model.model.layers -> base_model.model.layers
        if "base_model." not in k:
            return k
        # iterative replace to handle variable depths
        while "base_model.model.model." in k:
            k = k.replace("base_model.model.model.", "base_model.model.")
        return k

    # Rewrite weights for common adapter weight filenames
    # - safetensors: adapter_model.safetensors
    # - torch: adapter_model.bin
    wt_st = work_dir / "adapter_model.safetensors"
    wt_bin = work_dir / "adapter_model.bin"

    if wt_st.exists():
        try:
            from safetensors.torch import load_file, save_file

            sd = load_file(str(wt_st))
            new_sd = {}
            changed = 0
            for k, v in sd.items():
                nk = _normalize_key(k)
                new_sd[nk] = v
                if nk != k:
                    changed += 1
            if changed:
                save_file(new_sd, str(wt_st))
        except Exception:
            # If safetensors not available, fall back to leaving as-is.
            pass
    elif wt_bin.exists():
        try:
            import torch

            sd = torch.load(str(wt_bin), map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
                # sometimes wrapped
                raw = sd["state_dict"]
                wrap = True
            else:
                raw = sd
                wrap = False

            if isinstance(raw, dict):
                new_raw = {}
                changed = 0
                for k, v in raw.items():
                    nk = _normalize_key(k)
                    new_raw[nk] = v
                    if nk != k:
                        changed += 1
                if changed:
                    if wrap:
                        sd["state_dict"] = new_raw
                        torch.save(sd, str(wt_bin))
                    else:
                        torch.save(new_raw, str(wt_bin))
        except Exception:
            pass

    return work_dir



def _adapter_expected_in_features(adapter_dir: Path) -> Optional[int]:
    """Best-effort read of LoRA in_features from adapter weights.

    Used to detect obvious base-model mismatches early (e.g., adapter trained on 7B but base is 0.5B).

    IMPORTANT: Only use q_proj (or other attention projection that matches hidden_size).
    MLP layers like down_proj/up_proj have intermediate_size != hidden_size and would
    cause false mismatches.
    """
    wt_st = adapter_dir / "adapter_model.safetensors"
    wt_bin = adapter_dir / "adapter_model.bin"

    def _pick_shape(sd: Dict) -> Optional[int]:
        # Prefer q_proj / o_proj which always have in_features == hidden_size
        # Avoid k_proj/v_proj (GQA may differ) and MLP layers (intermediate_size)
        preferred = ["q_proj", "o_proj"]
        fallback_keys = []

        for k, v in sd.items():
            if not (isinstance(k, str) and "lora_A" in k and k.endswith("weight")):
                continue
            # Check if this is a preferred attention layer
            for pref in preferred:
                if pref in k:
                    try:
                        return int(v.shape[1])
                    except Exception:
                        pass
            fallback_keys.append((k, v))

        # If no preferred key found, skip — we can't reliably determine hidden_size
        # from MLP or GQA layers, so don't guess
        return None

    if wt_st.exists():
        try:
            from safetensors.torch import load_file
            sd = load_file(str(wt_st))
            return _pick_shape(sd)  # type: ignore[arg-type]
        except Exception:
            return None
    if wt_bin.exists():
        try:
            import torch
            sd = torch.load(str(wt_bin), map_location="cpu", weights_only=True)
            if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
                sd = sd["state_dict"]
            if isinstance(sd, dict):
                return _pick_shape(sd)
        except Exception:
            return None
    return None

def _is_peft_adapter_dir(p: Path) -> bool:
    """PEFT LoRA adapter folder typically contains adapter_config.json."""
    return p.is_dir() and (p / "adapter_config.json").exists()


def _is_full_model_dir(p: Path) -> bool:
    """HF model folder typically has config.json with a model_type key."""
    if not (p.is_dir() and (p / "config.json").exists()):
        return False
    try:
        import json
        cfg = json.loads((p / "config.json").read_text(encoding="utf-8"))
        return isinstance(cfg, dict) and bool(cfg.get("model_type"))
    except Exception:
        return False


class Merger:
    """MergeKit LoRA/模型合併器"""

    METHODS = ["slerp", "ties", "dare_ties", "dare_linear", "linear", "passthrough", "task_arithmetic"]

    def merge(
        self,
        base_model: str,
        loras: List[Dict],
        params: Dict,
        task: Optional[Task] = None,
    ) -> Path:
        """合併 LoRA 或模型

        注意：data/loras 目錄同時存放「PEFT LoRA 适配器」與「完整模型」。
        mergekit 的 YAML 合併主要面向完整模型；如果把 LoRA 适配器目錄當作模型丟進去，
        transformers 會在讀取 AutoConfig 時報：config.json 缺少 model_type。

        這裡做一層自動分流：
        - 若輸入全是 LoRA 适配器：走 PEFT merge_and_unload（支持多個适配器，順序合併）
        - 若輸入全是完整模型：走 mergekit
        - 若混用：直接報錯，避免產生「看似成功但結果不可用」的輸出
        """
        if not loras:
            raise ValueError("至少需要一個 LoRA/模型")

        method = params.get("method", "slerp")
        if method not in self.METHODS:
            raise ValueError(f"不支持的合併方法: {method}")

        merge_name = params.get("name", random_name("merged"))
        output_dir = LORAS_DIR / merge_name
        output_dir.mkdir(parents=True, exist_ok=True)

        progress_cb = task.update_progress if task else lambda p, m="": log(m)

        # ---- classify inputs (adapter vs full model) ----
        paths: List[Path] = []
        adapter_flags: List[bool] = []
        model_flags: List[bool] = []
        for l in loras:
            p = LORAS_DIR / l.get("filename", "")
            paths.append(p)
            adapter_flags.append(_is_peft_adapter_dir(p))
            model_flags.append(_is_full_model_dir(p))

        if all(adapter_flags):
            # PEFT adapters: merge into base model (supports multiple adapters by sequential merge)
            progress_cb(10, "檢測到 LoRA 适配器，使用 PEFT 合併到基礎模型...")
            out = self.merge_multiple_adapters_to_base(base_model, [str(p) for p in paths], merge_name, task=task)
            progress_cb(100, f"合併完成: {out}")
            return out

        if any(adapter_flags) and any(model_flags):
            raise ValueError(
                "混合輸入：你同時選了『LoRA 适配器』與『完整模型』。\n"
                "- LoRA 适配器（adapter_config.json）需要先用 PEFT 合併到基礎模型\n"
                "- 完整模型（config.json 含 model_type）才可直接用 MergeKit" 
            )

        # Default: treat as mergekit models
        if yaml is None:
            raise ImportError("MergeKit 合併需要 PyYAML。请执行: pip install pyyaml")
        progress_cb(10, f"生成 {method} 合併配置...")
        try:
            config_data = self._generate_config(base_model, loras, params)
            config_path = output_dir / "merge_config.yaml"
            config_path.write_text(
                yaml.dump(config_data, allow_unicode=True, default_flow_style=False),
                encoding="utf-8",
            )

            progress_cb(20, "啟動 MergeKit...")
            import sys
            cmd = [
                sys.executable, "-m", "mergekit.scripts.run_yaml",
                str(config_path),
                str(output_dir),
                "--copy-tokenizer",
                "--allow-crimes",
                "--out-shard-size", "5000M",
            ]
            if params.get("cuda", True):
                cmd.append("--cuda")

            run_subprocess(cmd, progress_cb=lambda line: progress_cb(50, line))

            # 复制 trust_remote_code 模型的自定义 .py 文件
            _copy_custom_model_code(base_model, output_dir)

            progress_cb(100, f"合併完成: {output_dir}")
            return output_dir
        except Exception as e:
            log(f"❌ 合併失敗: {e}")
            raise

    def _generate_config(self, base_model: str, loras: List[Dict], params: Dict) -> Dict:
        """生成正確的 MergeKit YAML 配置"""
        method = params.get("method", "slerp")

        # MergeKit 使用 models 列表（不是 slices）
        models = []

        # 基礎模型
        base_entry = {"model": base_model}

        if method == "slerp":
            # SLERP：只支持兩個模型，t 參數控制插值
            if len(loras) != 1:
                log("⚠️ SLERP 最佳用於兩個模型間，使用第一個 LoRA")

            lora_path = str(LORAS_DIR / loras[0]["filename"])
            models = [
                {"model": base_model, "parameters": {"weight": 1.0 - params.get("t", 0.5)}},
                {"model": lora_path, "parameters": {"weight": params.get("t", 0.5)}},
            ]
            config_data = {
                "merge_method": "slerp",
                "slices": [{
                    "sources": [
                        {"model": base_model, "layer_range": [0, 999]},
                        {"model": lora_path, "layer_range": [0, 999]},
                    ],
                    "parameters": {"t": params.get("t", 0.5)},
                }],
                "dtype": params.get("dtype", "bfloat16"),
            }
            return config_data

        elif method in ("ties", "dare_ties"):
            # TIES / DARE-TIES：多模型合併
            models = [{"model": base_model}]  # base 不帶參數
            for lora in loras:
                lora_path = str(LORAS_DIR / lora["filename"])
                models.append({
                    "model": lora_path,
                    "parameters": {
                        "density": params.get("density", 0.5),
                        "weight": lora.get("weight", 1.0),
                    },
                })
            return {
                "merge_method": method,
                "base_model": base_model,
                "models": models,
                "parameters": {
                    "normalize": params.get("normalize", True),
                    "int8_mask": params.get("int8_mask", True),
                },
                "dtype": params.get("dtype", "bfloat16"),
            }

        elif method in ("dare_linear", "linear"):
            models = [{"model": base_model}]
            for lora in loras:
                lora_path = str(LORAS_DIR / lora["filename"])
                models.append({
                    "model": lora_path,
                    "parameters": {
                        "weight": lora.get("weight", 1.0),
                        **({"density": params.get("density", 0.5)} if "dare" in method else {}),
                    },
                })
            return {
                "merge_method": method,
                "base_model": base_model,
                "models": models,
                "dtype": params.get("dtype", "bfloat16"),
            }

        elif method == "task_arithmetic":
            models = [{"model": base_model}]
            for lora in loras:
                lora_path = str(LORAS_DIR / lora["filename"])
                models.append({
                    "model": lora_path,
                    "parameters": {"weight": lora.get("weight", 1.0)},
                })
            return {
                "merge_method": "task_arithmetic",
                "base_model": base_model,
                "models": models,
                "dtype": params.get("dtype", "bfloat16"),
            }

        else:  # passthrough 等
            return {
                "merge_method": method,
                "slices": [{
                    "sources": [{"model": str(LORAS_DIR / l["filename"]), "layer_range": [0, 999]} for l in loras],
                }],
                "dtype": params.get("dtype", "bfloat16"),
            }

    def list_available_loras(self) -> List[Dict]:
        """掃描可用 LoRA"""
        loras = []
        if not LORAS_DIR.exists():
            return loras

        for p in LORAS_DIR.iterdir():
            if p.is_dir():
                # 檢查是否是有效的 LoRA 或模型目錄
                is_lora = (p / "adapter_config.json").exists()
                is_model = (p / "config.json").exists()
                if is_lora or is_model:
                    info = {"filename": p.name, "path": str(p), "type": "lora" if is_lora else "model"}
                    # 讀取 ForgeX 訓練信息
                    train_info_path = p / "forgex_train_info.json"
                    if train_info_path.exists():
                        try:
                            import json
                            info["train_info"] = json.loads(train_info_path.read_text(encoding="utf-8"))
                        except Exception:
                            pass
                    # 計算大小
                    try:
                        info["size"] = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                    except Exception:
                        info["size"] = 0
                    loras.append(info)
        return loras

    def merge_multiple_adapters_to_base(
        self,
        base_model: str,
        adapter_paths: List[str],
        output_name: str,
        task: Optional[Task] = None,
    ) -> Path:
        """將多個 LoRA 适配器順序合併到同一個基礎模型。

        做法（務實版）：
        - 先加载 base
        - 逐個加载 adapter → merge_and_unload
        - 每次 merge 後，模型就變成新的「基礎權重」

        這能穩定解決：把 adapter 目錄直接丟進 mergekit 導致的 AutoConfig 讀取失敗。
        """
        progress_cb = task.update_progress if task else lambda p, m="": log(m)

        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError as e:
            raise ImportError("需要: pip install peft transformers torch") from e

        out_dir = LORAS_DIR / output_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # 使用临时目录存放 offload/rewrite，避免污染输出
        work_dir = out_dir / "_merge_workspace"
        work_dir.mkdir(parents=True, exist_ok=True)
        offload_dir = work_dir / "offload"
        offload_dir.mkdir(parents=True, exist_ok=True)

        # 自动选择 dtype（优先 float16，A100/H100 用 bfloat16）
        dtype = torch.float16
        try:
            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability()
                if cap[0] >= 8:  # Ampere+
                    dtype = torch.bfloat16
        except Exception:
            pass

        progress_cb(15, f"加载基礎模型: {base_model} ({dtype})")
        # 预导入模型类（防止 LazyAutoMapping 失效）
        try:
            from core.safe_loader import ensure_model_importable
            ensure_model_importable(base_model)
        except Exception:
            pass
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_folder=str(offload_dir),
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

        # Read base hidden size for quick shape sanity-check (best-effort).
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
            base_hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
        except Exception:
            base_hidden = None

        total = max(1, len(adapter_paths))
        for i, ap in enumerate(adapter_paths, start=1):
            progress_cb(20 + (i - 1) * (50 / total), f"加载 LoRA ({i}/{total}): {ap}")

            # Preflight: adapter expected in_features should match base hidden_size (if available).
            try:
                exp_in = _adapter_expected_in_features(Path(ap))
                if base_hidden and exp_in and int(base_hidden) != int(exp_in):
                    raise RuntimeError(
                        "LoRA 适配器形狀不匹配（通常是選錯 base 或 adapter 不是這個模型訓練出來的）。\n"
                        f"- base_model: {base_model} (hidden_size={base_hidden})\n"
                        f"- adapter: {ap} (adapter_in_features={exp_in})\n"
                        "请改用與 LoRA 訓練時一致的 base_model（同家族、同大小、同版本）。"
                    )
            except Exception as e:
                # If we cannot read adapter shapes, continue; peft load will validate later.
                if isinstance(e, RuntimeError):
                    raise

            # Load adapter
            # After merge_and_unload(), model loses hf_device_map; handle both cases.
            has_device_map = bool(getattr(model, "hf_device_map", None))
            try:
                try:
                    kwargs = {}
                    if has_device_map:
                        kwargs["device_map"] = "auto"
                        kwargs["offload_folder"] = str(offload_dir)
                    peft_model = PeftModel.from_pretrained(model, ap, **kwargs)
                except TypeError:
                    # Older PEFT versions may not accept these kwargs
                    peft_model = PeftModel.from_pretrained(model, ap)
            except Exception as e:
                msg = str(e)
                # Hard mismatch: stop early with clearer info.
                if "size mismatch" in msg:
                    exp_in2 = None
                    try:
                        exp_in2 = _adapter_expected_in_features(Path(ap))
                    except Exception:
                        pass
                    raise RuntimeError(
                        "LoRA 适配器加载失敗（權重維度與 base_model 不一致）。\n"
                        f"- base_model: {base_model} (hidden_size={base_hidden})\n"
                        f"- adapter: {ap} (adapter_in_features={exp_in2})\n"
                        f"- 原始错误: {repr(e)}"
                    ) from e

                # KeyError / wrapper-hop mismatch: rewrite adapter keys on disk and retry once.
                rewritten_dir = work_dir / f"adapter_rewritten_{i}"
                try:
                    _rewrite_adapter_keys(Path(ap), rewritten_dir)
                    peft_model = _peft_from_pretrained_with_offload(model, str(rewritten_dir), offload_dir)
                except Exception as e2:
                    raise RuntimeError(
                        "LoRA 适配器加载失敗（常見原因：LoRA 不是用這個 base_model 訓練的，或模型架構不同）。\n"
                        f"- base_model (你選的): {base_model}\n"
                        f"- adapter (路徑): {ap}\n"
                        f"- 原始错误: {repr(e)}\n"
                        f"- 重寫 keys 後错误: {repr(e2)}"
                    ) from e2

            progress_cb(20 + i * (50 / total), f"合併 LoRA ({i}/{total})…")
            model = peft_model.merge_and_unload()

        progress_cb(75, "保存合併後模型…")
        model.save_pretrained(out_dir, safe_serialization=True)
        tokenizer.save_pretrained(out_dir)

        # 复制 trust_remote_code 模型的自定义 .py 文件
        _copy_custom_model_code(base_model, out_dir)

        # 清理工作目录
        try:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass

        # Record merge recipe for reproducibility.
        try:
            meta = {
                "type": "merge",
                "base_model": base_model,
                "adapters": adapter_paths,
                "output_dir": str(out_dir),
                "dtype": str(dtype),
            }
            (out_dir / "forgex_merge_recipe.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass

        progress_cb(100, "合併完成")
        return out_dir
    def merge_lora_to_base(self, base_model: str, lora_path: str, output_name: str, task: Optional[Task] = None) -> Path:
        """將 LoRA 合併到基礎模型（PEFT merge_and_unload）"""
        progress_cb = task.update_progress if task else lambda p, m="": log(m)
        progress_cb(10, "加载依賴...")

        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError as e:
            raise ImportError("需要: pip install peft transformers torch") from e

        output_dir = LORAS_DIR / output_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # 工作目录（不污染输出）
        work_dir = output_dir / "_merge_workspace"
        work_dir.mkdir(parents=True, exist_ok=True)
        offload_dir = work_dir / "offload"
        offload_dir.mkdir(parents=True, exist_ok=True)

        # 自动选择 dtype
        dtype = torch.float16
        try:
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                dtype = torch.bfloat16
        except Exception:
            pass

        progress_cb(20, f"加载基礎模型: {base_model} ({dtype})")
        try:
            from core.safe_loader import ensure_model_importable
            ensure_model_importable(base_model)
        except Exception:
            pass
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_folder=str(offload_dir),
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

        progress_cb(50, f"加载 LoRA: {lora_path}")
        try:
            model = _peft_from_pretrained_with_offload(model, lora_path, offload_dir)
        except Exception as e:
            tmp = work_dir / "rewrite_single"
            rewritten = _rewrite_adapter_keys(Path(lora_path), tmp)
            try:
                model = _peft_from_pretrained_with_offload(model, str(rewritten), offload_dir)
            except Exception as e2:
                cfg = _read_adapter_config(Path(lora_path))
                base_hint = cfg.get("base_model_name_or_path") or cfg.get("base_model_name") or "(unknown)"
                targets = cfg.get("target_modules") or "(unknown)"
                raise RuntimeError(
                    "LoRA 适配器加载失敗（常見原因：LoRA 不是用這個 base_model 訓練的，或模型架構不同）。\n"
                    f"- base_model (你選的): {base_model}\n"
                    f"- adapter (路徑): {lora_path}\n"
                    f"- adapter_config.base_model_name_or_path: {base_hint}\n"
                    f"- adapter_config.target_modules: {targets}\n"
                    f"- 原始错误: {repr(e)}\n"
                    f"- 重寫 keys 後错误: {repr(e2)}"
                ) from e2

        progress_cb(70, "合併權重...")
        model = model.merge_and_unload()

        progress_cb(85, "保存合併模型...")
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)

        # 复制 trust_remote_code 模型的自定义 .py 文件
        _copy_custom_model_code(base_model, output_dir)

        # 清理工作目录
        try:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass

        progress_cb(100, f"LoRA 合併完成: {output_dir}")
        return output_dir


# ════════════════════════════════════════════════════════════════
#  NativeMerger — 原生张量级融合（不需要 mergekit）
# ════════════════════════════════════════════════════════════════

class NativeMerger:
    """纯 PyTorch 实现的模型杂交融合

    支持:
      - slerp:          球面线性插值（两个模型）
      - linear:         加权平均（任意多个模型）
      - ties:           TIES-Merging（修剪 + 符号解冲 + 合并）
      - dare_ties:      DARE + TIES（随机丢弃 + TIES）
      - dare_linear:    DARE + Linear
      - task_arithmetic: 任务向量算术
      - frankenmerge:   层级 Frankenmerge（从不同模型取不同层）

    所有操作直接在 state_dict 上执行，不依赖 mergekit。
    不考虑内存约束: 同时加载所有模型到内存。
    """

    NATIVE_METHODS = [
        "slerp", "linear", "ties", "dare_ties", "dare_linear",
        "task_arithmetic", "frankenmerge",
    ]

    def merge(
        self,
        base_model: str,
        model_paths: List[str],
        method: str = "slerp",
        output_name: str = "native_merged",
        weights: Optional[List[float]] = None,
        t: float = 0.5,
        density: float = 0.5,
        normalize: bool = True,
        franken_spec: Optional[List[Dict]] = None,
        task: Optional[Task] = None,
    ) -> Path:
        """执行原生张量级融合

        Args:
            base_model:    基座模型路径/HF ID
            model_paths:   参与融合的模型路径列表
            method:        融合算法
            weights:       每个模型的权重（None = 等权）
            t:             SLERP 插值参数 (0~1)
            density:       TIES/DARE 密度参数 (0~1)
            normalize:     TIES 是否归一化
            franken_spec:  Frankenmerge 规格 [{"model_idx": 0, "layers": [0,5]}, ...]
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        progress_cb = task.update_progress if task else lambda p, m="": log(m)

        if method not in self.NATIVE_METHODS:
            raise ValueError(f"不支持: {method}. 可选: {self.NATIVE_METHODS}")

        # 加载基座
        progress_cb(5, f"加载基座模型: {base_model}")
        base_sd = self._load_state_dict(base_model)
        base_config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

        # 加载所有模型
        model_sds = []
        for i, mp in enumerate(model_paths):
            progress_cb(5 + 20 * (i + 1) / len(model_paths),
                        f"加载模型 {i+1}/{len(model_paths)}: {mp}")
            model_sds.append(self._load_state_dict(mp))

        if weights is None:
            weights = [1.0 / len(model_sds)] * len(model_sds)

        # 选择融合算法
        progress_cb(30, f"开始 {method} 融合...")

        if method == "slerp":
            if len(model_sds) != 1:
                raise ValueError("SLERP 需要恰好 1 个模型（与基座插值）")
            merged_sd = self._slerp_merge(base_sd, model_sds[0], t, progress_cb)

        elif method == "linear":
            merged_sd = self._linear_merge(base_sd, model_sds, weights, progress_cb)

        elif method == "ties":
            merged_sd = self._ties_merge(base_sd, model_sds, weights, density,
                                          normalize, progress_cb)
        elif method == "dare_ties":
            merged_sd = self._dare_merge(base_sd, model_sds, weights, density,
                                          normalize, use_ties=True, progress_cb=progress_cb)
        elif method == "dare_linear":
            merged_sd = self._dare_merge(base_sd, model_sds, weights, density,
                                          normalize, use_ties=False, progress_cb=progress_cb)
        elif method == "task_arithmetic":
            merged_sd = self._task_arithmetic(base_sd, model_sds, weights, progress_cb)

        elif method == "frankenmerge":
            merged_sd = self._frankenmerge(base_sd, model_sds, franken_spec,
                                            base_config, progress_cb)
        else:
            raise ValueError(f"未实现: {method}")

        # 保存
        progress_cb(85, "保存融合模型...")
        out_dir = LORAS_DIR / output_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # 用基座模型架构创建空模型，加载融合权重
        model = AutoModelForCausalLM.from_config(base_config, trust_remote_code=True)
        model.load_state_dict(merged_sd, strict=False)
        model.save_pretrained(out_dir, safe_serialization=True)
        tokenizer.save_pretrained(out_dir)

        # 复制自定义代码
        _copy_custom_model_code(base_model, out_dir)

        # 元信息
        import json
        meta = {
            "type": "native_merge", "method": method,
            "base_model": base_model, "models": model_paths,
            "weights": weights, "t": t, "density": density,
        }
        try:
            (out_dir / "forgex_merge_recipe.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False, default=str))
        except Exception:
            pass

        del model, merged_sd, base_sd, model_sds
        import gc; gc.collect()

        progress_cb(100, f"✅ {method} 融合完成: {out_dir.name}")
        return out_dir

    # ---- 加载 ----

    def _load_state_dict(self, model_path: str) -> Dict:
        """加载模型 state_dict（全精度，不考虑内存）"""
        import torch
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32,
            trust_remote_code=True, device_map="cpu",
        )
        sd = {k: v.clone() for k, v in model.state_dict().items()}
        del model
        import gc; gc.collect()
        return sd

    # ---- SLERP ----

    def _slerp_merge(self, sd_a, sd_b, t, progress_cb):
        """球面线性插值 — 在超球面上沿大圆弧插值"""
        import torch

        merged = {}
        keys = list(sd_a.keys())

        for i, key in enumerate(keys):
            if i % 50 == 0:
                progress_cb(30 + 50 * i / len(keys), f"SLERP {i}/{len(keys)}")

            a, b = sd_a[key], sd_b.get(key)
            if b is None or a.shape != b.shape:
                merged[key] = a
                continue

            merged[key] = self._slerp_tensor(a, b, t)

        return merged

    @staticmethod
    def _slerp_tensor(a, b, t):
        """对两个张量做 SLERP"""
        import torch

        a_flat = a.float().flatten()
        b_flat = b.float().flatten()

        # 归一化
        a_norm = a_flat / (a_flat.norm() + 1e-8)
        b_norm = b_flat / (b_flat.norm() + 1e-8)

        # 计算夹角
        dot = torch.clamp(torch.dot(a_norm, b_norm), -1.0, 1.0)
        omega = torch.acos(dot)

        if omega.abs() < 1e-6:
            # 几乎平行 → 线性插值
            result = (1 - t) * a_flat + t * b_flat
        else:
            sin_omega = torch.sin(omega)
            result = (torch.sin((1 - t) * omega) / sin_omega) * a_flat + \
                     (torch.sin(t * omega) / sin_omega) * b_flat

        return result.reshape(a.shape).to(a.dtype)

    # ---- Linear ----

    def _linear_merge(self, base_sd, model_sds, weights, progress_cb):
        """加权线性平均"""
        import torch
        merged = {}
        keys = list(base_sd.keys())

        for i, key in enumerate(keys):
            if i % 50 == 0:
                progress_cb(30 + 50 * i / len(keys), f"Linear {i}/{len(keys)}")

            tensors = [base_sd[key].float()]
            w = [1.0 - sum(weights)]
            for j, sd in enumerate(model_sds):
                if key in sd and sd[key].shape == base_sd[key].shape:
                    tensors.append(sd[key].float())
                    w.append(weights[j])

            result = sum(t * wi for t, wi in zip(tensors, w))
            merged[key] = result.to(base_sd[key].dtype)

        return merged

    # ---- TIES-Merging ----

    def _ties_merge(self, base_sd, model_sds, weights, density, normalize, progress_cb):
        """TIES: Trim → Elect Sign → Disjoint Merge"""
        import torch

        merged = {}
        keys = list(base_sd.keys())

        for i, key in enumerate(keys):
            if i % 50 == 0:
                progress_cb(30 + 50 * i / len(keys), f"TIES {i}/{len(keys)}")

            base_t = base_sd[key].float()

            # 1. 计算每个模型的任务向量 (delta)
            deltas = []
            for j, sd in enumerate(model_sds):
                if key in sd and sd[key].shape == base_t.shape:
                    deltas.append((sd[key].float() - base_t) * weights[j])
                else:
                    deltas.append(torch.zeros_like(base_t))

            if not deltas:
                merged[key] = base_sd[key]
                continue

            # 2. Trim: 只保留最大的 density% 参数
            trimmed = []
            for d in deltas:
                threshold = d.abs().quantile(1.0 - density).item()
                mask = d.abs() >= threshold
                trimmed.append(d * mask.float())

            # 3. Elect Sign: 多数投票决定每个参数的符号
            stacked = torch.stack(trimmed)
            sign_votes = (stacked > 0).float().sum(0) - (stacked < 0).float().sum(0)
            elected_sign = torch.sign(sign_votes)
            # 0 的位置取第一个非零 delta 的符号
            zero_mask = elected_sign == 0
            if zero_mask.any():
                elected_sign[zero_mask] = torch.sign(stacked[:, zero_mask].sum(0))[zero_mask]

            # 4. Disjoint Merge: 只取与多数符号一致的分量
            result = torch.zeros_like(base_t)
            count = torch.zeros_like(base_t)
            for d in trimmed:
                agree = torch.sign(d) == elected_sign
                result += d * agree.float()
                count += agree.float()

            if normalize:
                count = count.clamp(min=1)
                result = result / count

            merged[key] = (base_t + result).to(base_sd[key].dtype)

        return merged

    # ---- DARE ----

    def _dare_merge(self, base_sd, model_sds, weights, density, normalize,
                    use_ties, progress_cb):
        """DARE: 随机丢弃 delta 后再融合 (TIES 或 Linear)"""
        import torch

        # 先对每个模型的 delta 做随机 mask
        dare_sds = []
        for j, sd in enumerate(model_sds):
            dare_sd = {}
            for key in base_sd:
                if key in sd and sd[key].shape == base_sd[key].shape:
                    delta = sd[key].float() - base_sd[key].float()
                    # 随机保留 density 比例
                    mask = (torch.rand_like(delta) < density).float()
                    # rescale 保持期望值
                    rescaled = delta * mask / max(density, 1e-6)
                    dare_sd[key] = (base_sd[key].float() + rescaled).to(sd[key].dtype)
                else:
                    dare_sd[key] = base_sd.get(key, sd.get(key))
            dare_sds.append(dare_sd)

        if use_ties:
            return self._ties_merge(base_sd, dare_sds, weights, 1.0, normalize, progress_cb)
        else:
            return self._linear_merge(base_sd, dare_sds, weights, progress_cb)

    # ---- Task Arithmetic ----

    def _task_arithmetic(self, base_sd, model_sds, weights, progress_cb):
        """任务向量算术: base + Σ(w_i * (model_i - base))"""
        import torch

        merged = {}
        keys = list(base_sd.keys())

        for i, key in enumerate(keys):
            if i % 50 == 0:
                progress_cb(30 + 50 * i / len(keys), f"TaskArith {i}/{len(keys)}")

            base_t = base_sd[key].float()
            delta_sum = torch.zeros_like(base_t)

            for j, sd in enumerate(model_sds):
                if key in sd and sd[key].shape == base_t.shape:
                    delta_sum += weights[j] * (sd[key].float() - base_t)

            merged[key] = (base_t + delta_sum).to(base_sd[key].dtype)

        return merged

    # ---- Frankenmerge ----

    def _frankenmerge(self, base_sd, model_sds, spec, config, progress_cb):
        """层级 Frankenmerge — 从不同模型取不同层组装

        spec: [{"model_idx": 0, "layer_start": 0, "layer_end": 12},
               {"model_idx": 1, "layer_start": 12, "layer_end": 24}, ...]
        model_idx: -1 = base, 0+ = model_sds index
        """
        import torch, re

        if not spec:
            raise ValueError("frankenmerge 需要 franken_spec 层分配规格")

        merged = {}
        n_layers = config.num_hidden_layers

        # 先复制所有 non-layer 参数 from base
        layer_pattern = re.compile(r'\.layers\.(\d+)\.')
        for key in base_sd:
            if not layer_pattern.search(key):
                merged[key] = base_sd[key].clone()

        # 按规格组装层
        progress_cb(35, "Frankenmerge: 组装层...")
        out_layer_idx = 0
        for si, segment in enumerate(spec):
            src_idx = segment.get("model_idx", -1)  # -1 = base
            l_start = segment.get("layer_start", 0)
            l_end = segment.get("layer_end", n_layers)

            src_sd = base_sd if src_idx < 0 else model_sds[min(src_idx, len(model_sds) - 1)]

            for src_layer in range(l_start, l_end):
                progress_cb(35 + 45 * out_layer_idx / n_layers,
                            f"层 {out_layer_idx}: ← 模型{src_idx}[{src_layer}]")
                src_prefix = f".layers.{src_layer}."
                dst_prefix = f".layers.{out_layer_idx}."

                for key in src_sd:
                    if src_prefix in key:
                        new_key = key.replace(src_prefix, dst_prefix)
                        merged[new_key] = src_sd[key].clone()

                out_layer_idx += 1

        log(f"Frankenmerge: 组装了 {out_layer_idx} 层")
        return merged


native_merger = NativeMerger()


# 單例
merger = Merger()
