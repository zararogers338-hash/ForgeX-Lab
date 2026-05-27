from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return ForgeX project root folder (the folder containing main.py).

    Works regardless of current working directory.
    """
    here = Path(__file__).resolve()
    # .../ForgeX_v2/core/paths.py -> .../ForgeX_v2
    return here.parent.parent


def resolve_user_path(p: str | os.PathLike, *, base: Path | None = None, extra_roots: list[Path] | None = None) -> Path:
    """Resolve a user-provided path.

    - If absolute: return it as-is.
    - If relative: try base (default project_root), then extra_roots (e.g. data/).
    """
    if p is None:
        raise ValueError("path is None")
    p = Path(str(p)).expanduser()
    if p.is_absolute():
        return p

    base = base or project_root()
    cand = (base / p).resolve()
    if cand.exists():
        return cand

    if extra_roots:
        for r in extra_roots:
            r = Path(r)
            cand2 = (r / p).resolve()
            if cand2.exists():
                return cand2

    return (base / p).resolve()
