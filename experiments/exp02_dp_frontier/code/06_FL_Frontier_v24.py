#!/usr/bin/env python3
"""Legacy compatibility wrapper for the renamed Exp02 frontier script."""
from __future__ import annotations

import runpy
from pathlib import Path


TARGET = Path(__file__).with_name("02_FL_Frontier_v24.py")


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
