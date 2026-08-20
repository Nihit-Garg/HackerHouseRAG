"""
conftest.py — Shared pytest configuration.
Ensures src/ is on sys.path for all test files.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add src/ to path so tests can import modules directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
