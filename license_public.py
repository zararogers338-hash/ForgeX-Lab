from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import json
import platform
import uuid
from typing import Any, Dict

APP_NAME = "ForgeX"
APP_VERSION = "3.2-user"
SIGNATURE_ALG = "RSA-SHA256-PKCS1v1.5"
ISSUER_KEY_ID = "5daafe75a3a1d11d"
RSA_N = 23189147932623747252524123554812226385119180324645123165367867271744378818619297161772096419309664357053297689427696753695460514405786119712166606106576339847226657804497847116340165509735355726373427606755184801159785975709607804022024077141844889656785970766551385523840371323795581481931114736850260290559783226348847559127442021623136854910953026446817782945491614330724999200471754420656387643556573016835418569727235599727676574673210707849254370955551851249854484006963422664233153743001902409417303941026181412923578651961275951740452971384775587326147739328888665867842618889447060762655247684776060067244263
RSA_E = 65537
_DIGESTINFO_SHA256_PREFIX = bytes.fromhex("3031300d060960864801650304020105000420")


def machine_fingerprint() -> str:
    bits = [
        platform.system(),
        platform.machine(),
        platform.node(),
        hex(uuid.getnode()),
    ]
    payload = "|".join(bits).encode("utf-8", "ignore")
    return hashlib.sha256(payload).hexdigest()[:24]


def canonical_license_payload(data: Dict[str, Any]) -> bytes:
    clean = {k: v for k, v in data.items() if k != "signature"}
    return json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def payload_digest(data: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_license_payload(data)).hexdigest()


def _b64url_decode(s: str) -> bytes:
    pad = "=" * ((4 - len(s) % 4) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("ascii"))


def verify_signature(data: Dict[str, Any]) -> None:
    signature = str(data.get("signature", "")).strip()
    if not signature:
        raise ValueError("密钥缺少 signature")
    if data.get("alg") != SIGNATURE_ALG:
        raise ValueError("密钥签名算法不匹配")
    if str(data.get("key_id", "")).strip() != ISSUER_KEY_ID:
        raise ValueError("密钥签发者不匹配")

    sig_bytes = _b64url_decode(signature)
    k = (RSA_N.bit_length() + 7) // 8
    if len(sig_bytes) != k:
        raise ValueError("密钥签名字节长度异常")

    em = pow(int.from_bytes(sig_bytes, "big"), RSA_E, RSA_N).to_bytes(k, "big")
    digest = hashlib.sha256(canonical_license_payload(data)).digest()
    t = _DIGESTINFO_SHA256_PREFIX + digest
    expected = b"\x00\x01" + (b"\xff" * (k - len(t) - 3)) + b"\x00" + t
    if em != expected:
        raise ValueError("密钥签名无效")


def load_license_text(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    if not raw:
        raise ValueError("license.key 是空文件")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"license.key 不是合法 JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("license.key 顶层必须是 JSON 对象")
    return data


def verify_license(data: Dict[str, Any]) -> Dict[str, Any]:
    # 签名验证保留（确保提供的 license.key 合法）
    verify_signature(data)

    if data.get("product") != APP_NAME:
        raise ValueError("密钥产品名不匹配")

    # ========== 完全便携版修改：任何电脑都能运行 ==========
    # 1. 过期检查完全移除（无论日期如何都通过）
    # 2. 机器指纹绑定完全移除（任何电脑都视为有效）
    # 3. 不再输出任何警告，静默通过

    # 机器绑定逻辑已禁用，始终通过
    bound = "ANY"

    return {
        "product": data.get("product", APP_NAME),
        "customer": data.get("customer", "Unknown"),
        "edition": data.get("edition", "USER"),
        "expires": data.get("expires", ""),
        "machine": bound,
        "features": list(data.get("features", [])),
        "digest": payload_digest(data),
        "alg": data.get("alg", SIGNATURE_ALG),
        "key_id": data.get("key_id", ISSUER_KEY_ID),
    }