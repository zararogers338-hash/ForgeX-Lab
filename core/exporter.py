# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

# core/exporter.py - ForgeX v2.7 導出：GGUF / Ollama
# 修復：gguf 包版本不兼容 convert_hf_to_gguf.py 的 MistralTokenizerType 問題
import shutil
import os
import sys
from pathlib import Path
from typing import Optional

from core import LORAS_DIR, log, run_subprocess, get_timestamp
from core.task_queue import Task

GGUF_QUANT_TYPES = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"]


def _which(name: str) -> str | None:
    return shutil.which(name)


def _check_gguf_version() -> dict:
    """检测 gguf 包版本及兼容性"""
    info = {"installed": False, "version": "", "has_mistral": False, "path": ""}
    try:
        import gguf
        info["installed"] = True
        info["version"] = getattr(gguf, "__version__", "unknown")
        info["path"] = str(Path(gguf.__file__).parent)
        try:
            from gguf.vocab import MistralTokenizerType  # noqa: F401
            info["has_mistral"] = True
        except ImportError:
            info["has_mistral"] = False
    except ImportError:
        pass
    return info


def _try_upgrade_gguf(progress_cb=None) -> bool:
    """尝试升级 gguf 包以匹配最新 convert_hf_to_gguf.py"""
    if progress_cb:
        progress_cb(15, "尝试升级 gguf 包...")

    # 方法 1: pip install 最新 gguf from PyPI
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "gguf", "--quiet"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            import importlib
            try:
                import gguf
                importlib.reload(gguf)
                from gguf import vocab as _v
                importlib.reload(_v)
                from gguf.vocab import MistralTokenizerType  # noqa: F401
                log("✅ gguf 包升级成功（PyPI）")
                return True
            except ImportError:
                pass
    except Exception as e:
        log(f"PyPI 升级 gguf 失败: {e}")

    # 方法 2: 从 llama.cpp GitHub 安装最新 gguf-py
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "gguf @ git+https://github.com/ggml-org/llama.cpp.git@master#subdirectory=gguf-py",
             "--quiet", "--force-reinstall"],
            capture_output=True, text=True, timeout=180,
        )
        if result.returncode == 0:
            log("✅ gguf 包升级成功（llama.cpp GitHub）")
            return True
        log(f"GitHub 安装 gguf 失败: {result.stderr[:300]}")
    except Exception as e:
        log(f"GitHub 安装 gguf 失败: {e}")

    return False


def _download_convert_script(dest_dir: Path) -> Optional[Path]:
    """从 llama.cpp GitHub 下载最新 convert_hf_to_gguf.py"""
    url = "https://raw.githubusercontent.com/ggml-org/llama.cpp/refs/heads/master/convert_hf_to_gguf.py"
    dest = dest_dir / "convert_hf_to_gguf_latest.py"
    try:
        import urllib.request
        log(f"下载最新 convert_hf_to_gguf.py ...")
        urllib.request.urlretrieve(url, str(dest))
        if dest.exists() and dest.stat().st_size > 10000:
            log(f"✅ 下载成功: {dest}")
            return dest
    except Exception as e:
        log(f"下载 convert_hf_to_gguf.py 失败: {e}")
    return None


def _find_convert_scripts() -> list:
    """搜索所有可用的 convert_hf_to_gguf.py 脚本"""
    candidates = []

    # 1. llama.cpp 仓库（最稳定）
    for p in [Path.cwd() / "llama.cpp", Path.home() / "llama.cpp",
              Path("D:/llama.cpp"), Path("C:/llama.cpp")]:
        try:
            script = p / "convert_hf_to_gguf.py"
            if script.exists():
                candidates.append({
                    "path": script, "source": "llama.cpp repo",
                    "priority": 1, "has_gguf_py": (p / "gguf-py").exists(),
                })
        except Exception:
            pass

    # 2. llama-cpp-python 附带
    try:
        import importlib.util
        spec = importlib.util.find_spec("llama_cpp")
        if spec and spec.submodule_search_locations:
            pkg_dir = Path(list(spec.submodule_search_locations)[0])
            for p in pkg_dir.parent.rglob("convert_hf_to_gguf.py"):
                candidates.append({
                    "path": p, "source": "llama-cpp-python",
                    "priority": 3, "has_gguf_py": False,
                })
                break
    except Exception:
        pass

    # 3. site-packages/bin 目录
    for site_dir in [Path(sys.prefix) / "Lib" / "site-packages" / "bin",
                     Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "bin"]:
        try:
            script = site_dir / "convert_hf_to_gguf.py"
            if script.exists() and not any(c["path"] == script for c in candidates):
                candidates.append({
                    "path": script, "source": "site-packages/bin",
                    "priority": 3, "has_gguf_py": False,
                })
        except Exception:
            pass

    candidates.sort(key=lambda x: x["priority"])
    return candidates


def _find_quantize_bin() -> Optional[str]:
    """搜索 quantize / llama-quantize 二进制"""
    for name in ["llama-quantize", "llama-quantize.exe", "quantize", "quantize.exe"]:
        p = shutil.which(name)
        if p:
            return p
    for repo in [Path.cwd() / "llama.cpp", Path.home() / "llama.cpp",
                 Path("D:/llama.cpp"), Path("C:/llama.cpp")]:
        for sub in ["build/bin/llama-quantize", "build/bin/llama-quantize.exe",
                     "build/bin/quantize", "build/bin/quantize.exe",
                     "llama-quantize", "llama-quantize.exe"]:
            p = repo / sub
            if p.exists():
                return str(p)
    return None


class Exporter:
    """模型導出 - GGUF 量化 + Ollama"""

    def export_gguf(
        self,
        model_path: str = "",
        model_dir: str = "",
        quant_type: str = "Q4_K_M",
        quant: str = "",
        output_name: str = "",
        task: Optional[Task] = None,
    ) -> Path:
        progress_cb = task.update_progress if task else (lambda p, m="": log(m))

        model_path = model_path or model_dir
        quant_type = (quant or quant_type).upper()

        if quant_type not in GGUF_QUANT_TYPES:
            raise ValueError(f"不支持的量化類型: {quant_type}（支持: {', '.join(GGUF_QUANT_TYPES)}）")
        if not model_path:
            raise ValueError("请指定模型路徑")

        model_dir = Path(model_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"模型不存在: {model_path}")

        # ============================================================
        # 关键检测: 是 LoRA 适配器 还是 完整模型？
        # ============================================================
        is_lora = (model_dir / "adapter_config.json").exists()
        has_config = (model_dir / "config.json").exists()

        if is_lora and not has_config:
            # 这是一个 LoRA 适配器目录，不能直接转 GGUF
            # 需要先合并到基座模型
            progress_cb(3, "⚠️ 检测到 LoRA 适配器，需要先合并到基座模型...")

            # 从 adapter_config.json 读取原始基座模型
            base_model = None
            try:
                import json
                adapter_cfg = json.loads((model_dir / "adapter_config.json").read_text(encoding="utf-8"))
                base_model = adapter_cfg.get("base_model_name_or_path", "")
            except Exception:
                pass

            if not base_model:
                raise ValueError(
                    f"❌ 你选择的是 LoRA 适配器目录（{model_dir.name}），不能直接导出 GGUF。\n\n"
                    "LoRA 适配器只包含微调的差异权重（adapter_config.json + adapter_model.safetensors），\n"
                    "没有完整的模型权重（config.json），无法直接转换为 GGUF。\n\n"
                    "正确步骤:\n"
                    "  1. 先到「🚀 部署导出」→「🔀 LoRA 合并」\n"
                    "  2. 选择基座模型 + LoRA 适配器 → 点击合并\n"
                    "  3. 合并完成后，用合并后的模型目录来导出 GGUF\n\n"
                    "无法自动合并：adapter_config.json 中未记录基座模型路径。"
                )

            # 自动合并
            progress_cb(5, f"🔀 自动合并 LoRA 到基座模型: {base_model}")
            try:
                from core.merger import merger
                merge_name = output_name or f"{model_dir.name}_merged"
                merged_dir = merger.merge_lora_to_base(
                    base_model=base_model,
                    lora_path=str(model_dir),
                    output_name=merge_name + "_full",
                    task=task,
                )
                progress_cb(50, f"✅ 合并完成: {merged_dir}")
                model_dir = Path(merged_dir)
                log(f"LoRA 已自动合并，继续导出 GGUF: {model_dir}")
            except Exception as e:
                raise RuntimeError(
                    f"❌ 自动合并 LoRA 失败。\n\n"
                    f"LoRA 适配器: {model_path}\n"
                    f"基座模型（自动检测）: {base_model}\n"
                    f"错误: {e}\n\n"
                    "请手动操作:\n"
                    "  1. 到「🚀 部署导出」→「🔀 LoRA 合并」\n"
                    "  2. 手动选择正确的基座模型和 LoRA\n"
                    "  3. 合并后再导出 GGUF"
                ) from e

        elif not has_config:
            raise ValueError(
                f"❌ 目录 {model_dir.name} 中没有 config.json。\n"
                "这不是一个有效的 HuggingFace 模型目录。\n"
                "请选择一个完整的模型目录（合并后的模型或原始 HF 模型）。"
            )

        if not output_name:
            output_name = f"{model_dir.name}_{quant_type}_{get_timestamp()}"
        output_dir = LORAS_DIR / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        gguf_path = output_dir / f"{output_name}.gguf"

        progress_cb(5 if not is_lora else 55, "準備 GGUF 導出...")

        # ============================================================
        # 策略 1: llama.cpp 仓库脚本（最稳定，自带 gguf-py）
        # ============================================================
        try:
            progress_cb(8, "尝试 llama.cpp 仓库脚本...")
            return self._convert_with_llama_cpp_repo(model_dir, gguf_path, quant_type, progress_cb)
        except FileNotFoundError:
            log("llama.cpp 仓库未找到，尝试其他方式...")
        except Exception as e:
            log(f"llama.cpp 仓库脚本失败: {e}")

        # ============================================================
        # 策略 2: 检测 gguf 包兼容性 → 自动修复
        # ============================================================
        progress_cb(10, "检测 gguf 包兼容性...")
        gguf_info = _check_gguf_version()
        log(f"gguf 包状态: {gguf_info}")

        if gguf_info["installed"] and not gguf_info["has_mistral"]:
            progress_cb(12, "⚠️ gguf 包版本过旧，尝试自动升级...")
            upgraded = _try_upgrade_gguf(progress_cb)
            if upgraded:
                gguf_info = _check_gguf_version()
                progress_cb(18, "✅ gguf 包已升级")
            else:
                progress_cb(18, "⚠️ gguf 包升级失败，尝试其他方式...")

        # ============================================================
        # 策略 3: 使用已安装的 convert_hf_to_gguf.py
        # ============================================================
        scripts = _find_convert_scripts()
        for script_info in scripts:
            try:
                progress_cb(20, f"尝试: {script_info['source']} ...")
                return self._convert_with_script(
                    script_info["path"], script_info.get("has_gguf_py", False),
                    model_dir, gguf_path, quant_type, progress_cb,
                )
            except Exception as e:
                log(f"{script_info['source']} 失败: {str(e)[:200]}")
                continue

        # ============================================================
        # 策略 4: 从 GitHub 下载最新脚本
        # ============================================================
        try:
            progress_cb(25, "从 GitHub 下载最新转换脚本...")
            downloaded = _download_convert_script(output_dir)
            if downloaded:
                return self._convert_with_script(
                    downloaded, False,
                    model_dir, gguf_path, quant_type, progress_cb,
                )
        except Exception as e:
            log(f"GitHub 脚本失败: {e}")

        # ============================================================
        # 全部失败 → 给出详细修复指南
        # ============================================================
        gguf_info = _check_gguf_version()
        guide = self._build_fix_guide(gguf_info, scripts)

        # 收集所有尝试过的策略的最后错误
        last_errors = []
        if scripts:
            last_errors.append(f"已尝试 {len(scripts)} 个转换脚本（含自动补丁）")
        if gguf_info["installed"] and not gguf_info["has_mistral"]:
            last_errors.append(f"gguf v{gguf_info['version']} 缺少 MistralTokenizerType")

        raise RuntimeError(
            f"{guide}\n\n"
            f"{''.join(f'· {e}' + chr(10) for e in last_errors)}"
        )

    def _convert_with_script(
        self, script_path: Path, has_local_gguf_py: bool,
        model_dir: Path, gguf_path: Path, quant_type: str, progress_cb,
    ) -> Path:
        """使用指定的 convert_hf_to_gguf.py 脚本转换

        核心修复: 如果 gguf 包缺少 MistralTokenizerType，会自动创建
        包装脚本，在运行前注入缺失的符号。
        """
        f16_path = gguf_path.parent / f"{gguf_path.stem}_f16.gguf"

        env = os.environ.copy()
        if has_local_gguf_py:
            gguf_py_dir = script_path.parent / "gguf-py"
            if gguf_py_dir.exists():
                env["PYTHONPATH"] = str(gguf_py_dir) + os.pathsep + env.get("PYTHONPATH", "")

        # ----------------------------------------------------------
        # 核心: 当 gguf 包缺少 MistralTokenizerType 时，
        # 优先使用 wrapper 脚本（最可靠的方式）
        # ----------------------------------------------------------
        actual_script = script_path
        gguf_info = _check_gguf_version()
        needs_patch = gguf_info["installed"] and not gguf_info["has_mistral"] and not has_local_gguf_py

        if needs_patch:
            progress_cb(30, "⚠️ gguf 包缺少 MistralTokenizerType，创建兼容包装...")
            # 优先 wrapper（在 Python 级别注入，最可靠）
            wrapper = self._create_wrapper_script(script_path, gguf_path.parent)
            if wrapper:
                actual_script = wrapper
                log(f"使用包装脚本: {wrapper}")
            else:
                # 退回到文本补丁
                patched = self._create_patched_script(script_path, gguf_path.parent)
                if patched:
                    actual_script = patched
                    log(f"使用补丁脚本: {patched}")

        progress_cb(35, f"转换 → F16 GGUF ({actual_script.name})...")

        import subprocess
        result = subprocess.run(
            [sys.executable, str(actual_script), str(model_dir),
             "--outfile", str(f16_path), "--outtype", "f16"],
            capture_output=True, text=True, timeout=3600,
            env=env,
        )

        # 如果第一次失败，且错误仍是 MistralTokenizerType，尝试 Plan B
        if result.returncode != 0:
            error_text = (result.stderr + result.stdout)[-800:]

            if "MistralTokenizerType" in error_text or "MistralVocab" in error_text:
                # Plan B: 如果还没用 wrapper，现在用
                if actual_script != script_path or not needs_patch:
                    # wrapper/patch 也失败了，尝试另一种方式
                    progress_cb(37, "补丁失败，尝试备用包装脚本...")
                    alt = (self._create_wrapper_script if actual_script != self._create_wrapper_script
                           else self._create_patched_script)
                else:
                    alt = self._create_wrapper_script

                wrapper_b = self._create_wrapper_script(script_path, gguf_path.parent)
                if wrapper_b and wrapper_b != actual_script:
                    result = subprocess.run(
                        [sys.executable, str(wrapper_b), str(model_dir),
                         "--outfile", str(f16_path), "--outtype", "f16"],
                        capture_output=True, text=True, timeout=3600,
                        env=env,
                    )
                    if result.returncode != 0:
                        error_text = (result.stderr + result.stdout)[-800:]

                        # Plan C: 直接在子进程中预注入
                        progress_cb(38, "尝试内联注入方式...")
                        inject_result = self._convert_with_inline_inject(
                            script_path, model_dir, f16_path, env)
                        if inject_result and f16_path.exists():
                            progress_cb(60, "✅ 内联注入方式成功")
                        else:
                            raise RuntimeError(
                                f"GGUF 導出失敗（gguf 包版本不兼容）。\n\n"
                                f"根本原因: gguf v{gguf_info.get('version','?')} 缺少 MistralTokenizerType\n\n"
                                f"一步修复:\n"
                                f'  pip install "gguf @ git+https://github.com/ggml-org/llama.cpp.git@master#subdirectory=gguf-py"\n\n'
                                f"或克隆 llama.cpp 仓库:\n"
                                f"  git clone https://github.com/ggml-org/llama.cpp.git\n"
                                f"  cd llama.cpp && pip install -e gguf-py\n\n"
                                f"最後错误: {error_text[-300:]}")
                    else:
                        progress_cb(60, "✅ 备用包装脚本成功")
                else:
                    raise RuntimeError(
                        f"GGUF 導出失敗（gguf 包版本不兼容）。\n\n"
                        f"根本原因: gguf v{gguf_info.get('version','?')} 缺少 MistralTokenizerType\n\n"
                        f"一步修复:\n"
                        f'  pip install "gguf @ git+https://github.com/ggml-org/llama.cpp.git@master#subdirectory=gguf-py"\n\n'
                        f"最後错误: {error_text[-300:]}")
            else:
                raise RuntimeError(f"Command failed (rc={result.returncode}): {error_text}")

        if not f16_path.exists():
            raise FileNotFoundError(f"转换后文件不存在: {f16_path}")

        if quant_type == "F16":
            if gguf_path.exists():
                gguf_path.unlink()
            f16_path.rename(gguf_path)
            progress_cb(100, f"完成: {gguf_path}")
            return gguf_path

        progress_cb(70, f"量化 {quant_type}...")
        quant_bin = _find_quantize_bin()
        if quant_bin is None:
            progress_cb(90, f"⚠️ 未找到 quantize 工具，输出 F16")
            if gguf_path.exists():
                gguf_path.unlink()
            f16_path.rename(gguf_path)
            progress_cb(100, f"完成(F16): {gguf_path}")
            return gguf_path

        run_subprocess(
            [str(quant_bin), str(f16_path), str(gguf_path), quant_type],
            progress_cb=lambda l: progress_cb(90, l),
        )
        f16_path.unlink(missing_ok=True)
        progress_cb(100, f"完成: {gguf_path}")
        return gguf_path

    def _convert_with_inline_inject(self, script_path: Path, model_dir: Path,
                                      f16_path: Path, env: dict) -> bool:
        """Plan C: 用 -c 参数直接在子进程中注入缺失符号后执行脚本"""
        try:
            import subprocess
            code = f'''
import sys, enum, importlib

# 注入 MistralTokenizerType
class _MistralTokenizerType(enum.Enum):
    spm = "spm"; tekken = "tekken"
class _MistralVocab:
    def __init__(self, *a, **kw): raise NotImplementedError("MistralVocab stub")

try:
    from gguf import vocab as _gv
    if not hasattr(_gv, "MistralTokenizerType"):
        _gv.MistralTokenizerType = _MistralTokenizerType
    if not hasattr(_gv, "MistralVocab"):
        _gv.MistralVocab = _MistralVocab
except Exception:
    pass

sys.argv = [r"{script_path}", r"{model_dir}", "--outfile", r"{f16_path}", "--outtype", "f16"]
with open(r"{script_path}", "r", encoding="utf-8") as _f:
    exec(compile(_f.read(), r"{script_path}", "exec"), {{"__name__":"__main__","__file__":r"{script_path}"}})
'''
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, timeout=3600, env=env,
            )
            return result.returncode == 0 and f16_path.exists()
        except Exception as e:
            log(f"内联注入失败: {e}")
            return False

    def _create_patched_script(self, original: Path, work_dir: Path) -> Optional[Path]:
        """创建补丁版本的 convert_hf_to_gguf.py，将 MistralTokenizerType 导入包裹在 try/except 中"""
        try:
            import re
            src = original.read_text(encoding="utf-8")

            # 用正则匹配各种可能的 import 形式
            # 匹配: from gguf.vocab import ... MistralTokenizerType ...
            pattern = r'^(from\s+gguf\.vocab\s+import\s+.*MistralTokenizerType.*)$'
            match = re.search(pattern, src, re.MULTILINE)
            if not match:
                # 也尝试: from gguf import ... (某些版本的写法)
                pattern2 = r'^(from\s+gguf\b.*import\s+.*MistralTokenizerType.*)$'
                match = re.search(pattern2, src, re.MULTILINE)

            if not match:
                log("补丁脚本: 未找到 MistralTokenizerType import 语句")
                return None

            old_import = match.group(1)
            safe_import = f'''
# [ForgeX patch] 安全导入 MistralTokenizerType（兼容旧版 gguf）
try:
    {old_import}
except ImportError:
    import enum as _enum
    class MistralTokenizerType(_enum.Enum):
        spm = "spm"
        tekken = "tekken"
    class MistralVocab:
        def __init__(self, *a, **kw):
            raise NotImplementedError(
                "MistralVocab 需要最新 gguf 包。"
                "此模型不使用 Mistral tokenizer，不影响转换。"
            )
    # 注入回 gguf.vocab 模块
    try:
        from gguf import vocab as _gv
        if not hasattr(_gv, "MistralTokenizerType"):
            _gv.MistralTokenizerType = MistralTokenizerType
        if not hasattr(_gv, "MistralVocab"):
            _gv.MistralVocab = MistralVocab
    except Exception:
        pass
'''
            patched_src = src.replace(old_import, safe_import, 1)

            dest = work_dir / "convert_hf_to_gguf_patched.py"
            dest.write_text(patched_src, encoding="utf-8")
            log(f"✅ 已创建补丁脚本: {dest}")
            return dest
        except Exception as e:
            log(f"创建补丁脚本失败: {e}")
            return None

    def _create_wrapper_script(self, original: Path, work_dir: Path) -> Optional[Path]:
        """创建包装脚本: 先注入缺失符号到 gguf.vocab，然后 exec 原始脚本"""
        try:
            wrapper_code = f'''#!/usr/bin/env python3
# [ForgeX] Wrapper: 注入 MistralTokenizerType 后运行 convert_hf_to_gguf.py
import sys, enum, importlib

# Step 1: 确保 gguf.vocab 有 MistralTokenizerType
try:
    from gguf.vocab import MistralTokenizerType
except ImportError:
    class MistralTokenizerType(enum.Enum):
        spm = "spm"
        tekken = "tekken"

    class MistralVocab:
        def __init__(self, *a, **kw):
            raise NotImplementedError("MistralVocab 需要最新 gguf 包")

    try:
        from gguf import vocab as _gv
        _gv.MistralTokenizerType = MistralTokenizerType
        _gv.MistralVocab = MistralVocab
    except Exception:
        pass

    # 也注入到 sys.modules 以防 from X import Y 再次触发
    import types
    if "gguf.vocab" not in sys.modules:
        mod = types.ModuleType("gguf.vocab")
        mod.MistralTokenizerType = MistralTokenizerType
        mod.MistralVocab = MistralVocab
        sys.modules["gguf.vocab"] = mod
    else:
        m = sys.modules["gguf.vocab"]
        if not hasattr(m, "MistralTokenizerType"):
            m.MistralTokenizerType = MistralTokenizerType
        if not hasattr(m, "MistralVocab"):
            m.MistralVocab = MistralVocab

# Step 2: exec 原始脚本
script_path = r"{str(original)}"
sys.argv[0] = script_path
with open(script_path, "r", encoding="utf-8") as _f:
    _code = _f.read()
exec(compile(_code, script_path, "exec"), {{"__name__": "__main__", "__file__": script_path}})
'''
            dest = work_dir / "convert_hf_to_gguf_wrapper.py"
            dest.write_text(wrapper_code, encoding="utf-8")
            log(f"✅ 已创建包装脚本: {dest}")
            return dest
        except Exception as e:
            log(f"创建包装脚本失败: {e}")
            return None

    def _find_llama_cpp_repo(self) -> Optional[Path]:
        for p in [Path.cwd() / "llama.cpp", Path.home() / "llama.cpp",
                  Path("D:/llama.cpp"), Path("C:/llama.cpp")]:
            try:
                if (p / "convert_hf_to_gguf.py").exists():
                    return p
            except Exception:
                pass
        return None

    def _convert_with_llama_cpp_repo(self, model_dir, gguf_path, quant_type, progress_cb) -> Path:
        repo = self._find_llama_cpp_repo()
        if repo is None:
            raise FileNotFoundError("未找到 llama.cpp 仓库")
        return self._convert_with_script(
            repo / "convert_hf_to_gguf.py", has_local_gguf_py=True,
            model_dir=model_dir, gguf_path=gguf_path,
            quant_type=quant_type, progress_cb=progress_cb,
        )

    def _build_fix_guide(self, gguf_info: dict, scripts: list) -> str:
        """构建详细的修复指南"""
        lines = ["GGUF 導出失敗。"]

        if not gguf_info["installed"]:
            lines.append("")
            lines.append("❌ 未安装 gguf 包")
            lines.append("   修复: pip install gguf")
        elif not gguf_info["has_mistral"]:
            lines.append("")
            lines.append(f"❌ gguf 包版本过旧 (v{gguf_info['version']}) — 缺少 MistralTokenizerType")
            lines.append("   convert_hf_to_gguf.py 需要最新版 gguf 包，但你安装的版本太旧。")
            lines.append("")
            lines.append("   最快修复（任选一）:")
            lines.append('   方案A: pip install "gguf @ git+https://github.com/ggml-org/llama.cpp.git@master#subdirectory=gguf-py"')
            lines.append("   方案B: pip install --upgrade gguf")
        else:
            lines.append("")
            lines.append(f"gguf 包正常 (v{gguf_info['version']})")

        if not scripts:
            lines.append("")
            lines.append("❌ 未找到 convert_hf_to_gguf.py 脚本")
            lines.append("   修复: pip install llama-cpp-python")

        lines.append("")
        lines.append("推荐一步到位（最稳定）:")
        lines.append("   git clone https://github.com/ggml-org/llama.cpp.git")
        lines.append("   cd llama.cpp && pip install -e gguf-py")
        if sys.platform == "win32":
            lines.append("   cmake -B build && cmake --build build --config Release")
        else:
            lines.append("   cmake -B build -DGGML_CUDA=ON && cmake --build build -j$(nproc)")
        lines.append("")
        lines.append("   ForgeX 会自动发现 llama.cpp 仓库并使用其脚本。")

        return "\n".join(lines)

    # ================================================================
    #  Ollama
    # ================================================================

    def export_ollama(
        self,
        model_path: str = "",
        gguf_path: str = "",
        ollama_name: str = "",
        model_name: str = "",
        quant_type: str = "Q4_K_M",
        system_prompt: str = "",
        task: Optional[Task] = None,
    ) -> dict:
        model_path = model_path or gguf_path
        ollama_name = ollama_name or model_name
        if not model_path:
            raise ValueError("请指定模型路徑")
        if not ollama_name:
            raise ValueError("请指定 Ollama 模型名")

        progress_cb = task.update_progress if task else (lambda p, m="": log(m))

        gp = Path(model_path)
        if gp.suffix.lower() == ".gguf" and gp.exists():
            gguf_file = gp
        else:
            gguf_file = self.export_gguf(model_path=model_path, quant_type=quant_type,
                                         output_name=ollama_name, task=task)

        mf = gguf_file.parent / "Modelfile"
        sys_block = f"SYSTEM {system_prompt}\n" if system_prompt else ""
        mf.write_text(f"FROM {gguf_file.name}\n{sys_block}\n", encoding="utf-8")
        progress_cb(90, f"已生成 Modelfile: {mf}")

        ollama_bin = shutil.which("ollama") or shutil.which("ollama.exe")
        if not ollama_bin:
            progress_cb(100, "⚠️ 未找到 ollama")
            return {
                "gguf_path": str(gguf_file), "modelfile": str(mf),
                "ollama_name": ollama_name, "status": "ollama_not_found",
                "manual_cmd": f"ollama create {ollama_name} -f {mf}",
            }
        try:
            run_subprocess([ollama_bin, "create", ollama_name, "-f", str(mf)],
                          progress_cb=lambda l: progress_cb(95, l))
            progress_cb(100, f"✅ 已導入 Ollama: {ollama_name}")
            return {"gguf_path": str(gguf_file), "modelfile": str(mf),
                    "ollama_name": ollama_name, "status": "imported"}
        except Exception as e:
            progress_cb(100, f"⚠️ Ollama 導入失敗: {e}")
            return {"gguf_path": str(gguf_file), "modelfile": str(mf),
                    "ollama_name": ollama_name, "status": f"import_failed: {e}",
                    "manual_cmd": f"ollama create {ollama_name} -f {mf}"}

    def export_ollama_modelfile(self, gguf_path: str, model_name: str = "",
                                task: Optional[Task] = None) -> Path:
        progress_cb = task.update_progress if task else (lambda p, m="": log(m))
        gp = Path(gguf_path)
        if not gp.exists():
            raise FileNotFoundError(f"GGUF 不存在: {gguf_path}")
        if not model_name:
            model_name = gp.stem
        mf = gp.parent / "Modelfile"
        mf.write_text(f"FROM {gp.name}\n\n", encoding="utf-8")
        progress_cb(100, f"已生成 Modelfile: {mf}")
        return mf


exporter = Exporter()
