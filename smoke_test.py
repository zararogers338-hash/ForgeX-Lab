"""Quick smoke tests for ForgeX packaging.

Run:
  python smoke_test.py

This does NOT require GPU; it only checks importability and basic wiring.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def main() -> int:
    import core
    from core.trainer import Trainer
    from core.merger import Merger
    from core.exporter import Exporter
    from core.dataset_manager import DatasetManager
    from core.utils import safe_json_save, safe_json_load

    # Basic instantiation
    Trainer()
    Merger()
    Exporter()
    dm = DatasetManager()
    # JSON helpers should never crash
    p = ROOT / "data" / "tmp" / "smoke.json"
    ok = safe_json_save(p, {"x": 1, "p": p})
    assert ok
    data = safe_json_load(p, {})
    assert data.get("x") == 1
    print("[OK] smoke tests passed")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
