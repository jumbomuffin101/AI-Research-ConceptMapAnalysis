"""Bundled reference sets for concept-map grading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REFERENCE_DIR = Path(__file__).resolve().parent
JULIA_PARKER_REFERENCE_PATH = REFERENCE_DIR / "julia_parker_reference.json"


def load_julia_parker_reference() -> dict[str, Any]:
    """Load the built-in Julia Parker Week 6 comparison reference set."""
    data = json.loads(JULIA_PARKER_REFERENCE_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Julia Parker reference material must be a JSON object.")
    return data
