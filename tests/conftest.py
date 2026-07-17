"""Test configuration for the source-layout package."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIRECTORY))

