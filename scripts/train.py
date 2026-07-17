"""Compatibility wrapper for the installed ``picotron`` command."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picotron.cli import main


if __name__ == "__main__":
    main()
