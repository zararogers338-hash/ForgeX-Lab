from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import hmac
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from license_public import machine_fingerprint, load_license_text, verify_license
except Exception:  # pragma: no cover - license module is legacy/optional in OSS build
    load_license_text = None
    verify_license = None

    def machine_fingerprint() -> str:
        bits = [platform.system(), platform.machine(), platform.node(), hex(uuid.getnode())]
        payload = "|".join(bits).encode("utf-8", "ignore")
        return hashlib.sha256(payload).hexdigest()[:24]


APP_NAME = "ForgeX"
APP_VERSION = "3.2.0-open-source"
ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
RUNTIME_DIR = ROOT / ".runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
RUNTIME_STATE = RUNTIME_DIR / "runtime_state.json"
DEFAULT_LICENSE = ROOT / "license.key"

_SECRET_PARTS = (
    "ForgeX", "::", "open-source-release", "::", "2026", "::", "launch-guard",
)


def eprint(msg: str = "") -> None:
    print(msg, file=sys.stderr)


def info(msg: str = "") -> None:
    print(msg)


def _secret() -> bytes:
    return hashlib.sha256("".join(_SECRET_PARTS).encode("utf-8")).digest()


def choose_bootstrap_python() -> str:
    if sys.executable and Path(sys.executable).exists():
        return sys.executable
    for candidate in ("python3", "python", "py"):
        p = shutil.which(candidate)
        if p:
            return p
    raise RuntimeError("Python 3.10+ was not found / 未找到可用 Python 3.10+")


def venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _onerror(func, path, exc_info):
    import stat
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def is_venv_broken(py_path: Path) -> bool:
    if not py_path.exists():
        return True
    try:
        result = subprocess.run([str(py_path), "--version"], capture_output=True, text=True, timeout=8)
        return result.returncode != 0
    except Exception:
        return True


def ensure_venv() -> Path:
    py = venv_python()
    if py.exists() and not is_venv_broken(py):
        return py

    if VENV_DIR.exists():
        info("[ForgeX] Detected a broken virtual environment; cleaning .venv ... / 检测到虚拟环境损坏，正在清理 .venv ...")
        try:
            shutil.rmtree(VENV_DIR, onerror=_onerror)
        except Exception:
            pass

    info("[ForgeX] Creating isolated virtual environment .venv / 正在创建独立虚拟环境 .venv")
    bootstrap = choose_bootstrap_python()
    result = subprocess.run([bootstrap, "-m", "venv", str(VENV_DIR)], capture_output=True, text=True)

    if result.returncode != 0:
        raise SystemExit(
            "Failed to create virtual environment / 虚拟环境创建失败。\n"
            "Try deleting .venv manually, or move the project to a short writable path such as C:\\ForgeX.\n"
            "建议手动删除 .venv 后重试，或把项目移动到 C:\\ForgeX 等短路径目录。\n"
            f"Error / 错误: {result.stderr}"
        )

    if not py.exists() or is_venv_broken(py):
        raise SystemExit("Virtual environment creation failed / 虚拟环境创建失败，请手动删除 .venv 后重试")

    return py


def ensure_python_version() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("ForgeX requires Python >= 3.10 / ForgeX 要求 Python >= 3.10")


def run_cmd(cmd: list[str], env: Optional[Dict[str, str]] = None, check: bool = True) -> int:
    pretty = " ".join(shlex.quote(x) for x in cmd)
    info(f"[cmd] {pretty}")
    proc = subprocess.run(cmd, env=env)
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc.returncode


def detect_nvidia_gpu() -> Optional[Dict[str, str]]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    for cmd in [
        [smi, "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
        [smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
    ]:
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip().splitlines()
            if out:
                parts = [p.strip() for p in out[0].split(",")]
                return {"name": parts[0], "driver": parts[1] if len(parts) > 1 else "", "compute_cap": parts[2] if len(parts) > 2 else ""}
        except Exception:
            continue
    return None


def torch_profile() -> Dict[str, Any]:
    gpu = detect_nvidia_gpu()
    if gpu:
        name = gpu.get("name", "")
        cap = gpu.get("compute_cap", "")
        try:
            if float(cap) >= 12.0 or any(x in name for x in ["RTX 50", "Blackwell"]):
                return {
                    "label": f"NVIDIA Blackwell / cu128 ({name})",
                    "args": ["torch==2.10.0", "torchvision==0.25.0", "torchaudio==2.10.0", "--index-url", "https://download.pytorch.org/whl/cu128"],
                    "kind": "nvidia-cu128",
                }
        except Exception:
            pass
        return {
            "label": f"NVIDIA CUDA / cu126 ({name})",
            "args": ["torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu126"],
            "kind": "nvidia-cu126",
        }
    return {"label": "CPU", "args": ["torch", "torchvision", "torchaudio"], "kind": "cpu"}


def read_runtime_state() -> Dict[str, Any]:
    if not RUNTIME_STATE.exists():
        return {}
    try:
        return json.loads(RUNTIME_STATE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_runtime_state(payload: Dict[str, Any]) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME_STATE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _runtime_deps_ready(py: Path) -> bool:
    if not py.exists():
        return False
    probe = r"""
import importlib.util, sys
from importlib.metadata import version
mods = ['gradio','torch','transformers','datasets','peft','trl','accelerate','huggingface_hub','numpy','pandas','yaml','PIL','fastapi','starlette']
missing = [m for m in mods if importlib.util.find_spec(m) is None]
bad = []
try:
    gv = version('gradio')
    fv = tuple(int(x) for x in version('fastapi').split('.')[:2])
    sv = tuple(int(x) for x in version('starlette').split('.')[:2])
    if not gv.startswith('4.44.1'):
        bad.append('gradio=' + gv)
    if fv >= (0, 120):
        bad.append('fastapi=' + version('fastapi'))
    if sv >= (1, 0):
        bad.append('starlette=' + version('starlette'))
except Exception as e:
    bad.append('version_check=' + repr(e))
print(','.join(missing + bad))
sys.exit(1 if missing or bad else 0)
    """
    try:
        result = subprocess.run([str(py), "-c", probe], capture_output=True, text=True, timeout=20)
        if result.returncode == 0:
            return True
        missing = (result.stdout or result.stderr or "").strip()
        if missing:
            info(f"[ForgeX] Missing or incompatible runtime deps / 运行环境缺少或不兼容: {missing}")
        return False
    except Exception:
        return False


def install_runtime(force: bool = False) -> None:
    ensure_python_version()
    py = ensure_venv()
    state = read_runtime_state()
    profile = torch_profile()
    req_mtime = max((ROOT / "requirements.txt").stat().st_mtime, (ROOT / "constraints.txt").stat().st_mtime)

    if (not force and state.get("profile") == profile["kind"] and
            state.get("version") == APP_VERSION and
            state.get("requirements_mtime") == req_mtime and
            _runtime_deps_ready(py)):
        info(f"[ForgeX] Runtime ready / 运行环境已就绪: {profile['label']}")
        return

    info(f"[ForgeX] Installing/updating runtime / 安装或更新运行环境: {profile['label']}")
    run_cmd([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run_cmd([str(py), "-m", "pip", "install", *profile["args"]])
    run_cmd([str(py), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt"), "-c", str(ROOT / "constraints.txt")])

    write_runtime_state({
        "version": APP_VERSION,
        "profile": profile["kind"],
        "requirements_mtime": req_mtime,
        "updated_at": _dt.datetime.now().isoformat(),
    })
    info("[ForgeX] Runtime installation complete / 运行环境装配完成")


def license_path_from_args(arg_path: Optional[str]) -> Path:
    if arg_path:
        return Path(arg_path).expanduser().resolve()
    env_path = os.environ.get("FORGEX_LICENSE_PATH", "").strip()
    return Path(env_path).expanduser().resolve() if env_path else DEFAULT_LICENSE


def open_source_meta() -> Dict[str, Any]:
    digest = hashlib.sha256(f"{APP_NAME}:{APP_VERSION}:open-source".encode()).hexdigest()
    return {
        "product": APP_NAME,
        "customer": "Open Source User",
        "edition": "OPEN_SOURCE",
        "expires": "never",
        "machine": "ANY",
        "features": ["ui", "train", "export", "dataset", "distill", "multimodal"],
        "digest": digest,
        "alg": "none",
        "key_id": "open-source",
    }


def ensure_license(path: Path, required: bool = False) -> Dict[str, Any]:
    """Legacy commercial-license compatibility.

    The open-source build does not require a license.key. If a license.key is present,
    this function will verify and report it; otherwise ForgeX starts in OPEN_SOURCE mode.
    """
    if not path.exists():
        if required:
            raise SystemExit(f"license.key not found / 未找到 license.key: {path}")
        info("[ForgeX] No license.key found; starting open-source build. / 未检测到 license.key，按开源版启动。")
        return open_source_meta()
    if load_license_text is None or verify_license is None:
        if required:
            raise SystemExit("license_public.py is unavailable / license_public.py 不可用")
        return open_source_meta()
    data = load_license_text(path.read_text(encoding="utf-8"))
    meta = verify_license(data)
    info(f"[ForgeX] License verified / 密钥校验通过: customer={meta.get('customer')} edition={meta.get('edition')}")
    return meta


def build_launch_token(session_id: str, digest: str) -> str:
    msg = f"{session_id}|{digest}|{machine_fingerprint()}".encode("utf-8")
    return hmac.new(_secret(), msg, hashlib.sha256).hexdigest()


def launch_main(meta: Dict[str, Any], extra_args: list[str]) -> int:
    py = ensure_venv()
    session_id = hashlib.sha256(os.urandom(32)).hexdigest()[:24]
    digest = meta.get("digest", "open-source")
    token = build_launch_token(session_id, digest)
    env = os.environ.copy()
    env.update({
        "FORGEX_SESSION_ID": session_id,
        "FORGEX_LICENSE_DIGEST": digest,
        "FORGEX_LAUNCH_TOKEN": token,
        "FORGEX_EDITION": meta.get("edition", "OPEN_SOURCE"),
    })
    cmd = [str(py), "-m", "main"] + extra_args
    return subprocess.call(cmd, env=env, cwd=str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="ForgeX open-source launcher / ForgeX 开源版启动器")
    parser.add_argument("command", nargs="?", default="run", choices=["run", "install", "fingerprint", "license-info"])
    parser.add_argument("--license", help="Path to legacy license.key / 旧版 license.key 路径")
    parser.add_argument("--force-install", action="store_true", help="Reinstall runtime / 强制重装运行环境")
    args, extra = parser.parse_known_args()

    if args.command == "fingerprint":
        print(machine_fingerprint())
        return

    ensure_python_version()

    if args.command == "install":
        install_runtime(force=args.force_install)
        return

    license_path = license_path_from_args(args.license)

    if args.command == "license-info":
        meta = ensure_license(license_path, required=False)
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        return

    if args.command == "run":
        meta = ensure_license(license_path, required=False)
        install_runtime(force=args.force_install)
        raise SystemExit(launch_main(meta, extra))


if __name__ == "__main__":
    main()
