"""Project-root path resolution for config and CLI paths (no hard-coded cwd)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_project_path(path_str: str, project_root: Optional[Path] = None) -> Path:
    """
    If path_str is absolute, return it resolved; otherwise resolve under project root.
    """
    root = project_root or PROJECT_ROOT
    p = Path(path_str.strip())
    return p.resolve() if p.is_absolute() else (root / p).resolve()
