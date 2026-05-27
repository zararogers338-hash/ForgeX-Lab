# ui/app.py
# Thin wrapper: keep a single source of truth for UI in main.py
from __future__ import annotations

from pathlib import Path
import sys

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import app  # main.py constructs the Gradio app as `app`

__all__ = ["app"]
