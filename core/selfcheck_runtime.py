# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

from .paths import project_root


def runtime_info() -> Dict[str, Any]:
    import platform
    info: Dict[str, Any] = {}
    info["python"] = sys.executable
    info["python_version"] = sys.version.split()[0]
    info["platform"] = platform.platform()
    info["project_root"] = str(project_root())
    try:
        import torch
        info["torch"] = getattr(torch, "__version__", "?")
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        if info["cuda_available"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as e:
        info["torch_error"] = str(e)
    return info
