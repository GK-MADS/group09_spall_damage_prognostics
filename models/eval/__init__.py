"""Evaluation and failure analysis for the spall models.

Design: docs/superpowers/specs/2026-10-09-model-evaluation-design.md
"""

import sys
from pathlib import Path

_MODELS = str(Path(__file__).resolve().parents[1])
if _MODELS not in sys.path:
    sys.path.insert(0, _MODELS)
