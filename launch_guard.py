from __future__ import annotations

import hashlib
import hmac
import platform
import uuid

_SECRET_PARTS = (
    "ForgeX",
    "::",
    "open-source-release",
    "::",
    "2026",
    "::",
    "launch-guard",
)


def _secret() -> bytes:
    raw = "".join(_SECRET_PARTS).encode("utf-8")
    return hashlib.sha256(raw).digest()


def machine_fingerprint() -> str:
    bits = [
        platform.system(),
        platform.machine(),
        platform.node(),
        hex(uuid.getnode()),
    ]
    payload = "|".join(bits).encode("utf-8", "ignore")
    return hashlib.sha256(payload).hexdigest()[:24]


def build_launch_token(session_id: str, license_digest: str = "open-source") -> str:
    msg = f"{session_id}|{license_digest}|{machine_fingerprint()}".encode("utf-8")
    return hmac.new(_secret(), msg, hashlib.sha256).hexdigest()


def require_user_launch() -> None:
    """Open-source build: direct `python main.py` startup is allowed."""
    return
