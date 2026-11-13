"""Ensure the GUI source directory is importable when pytest is launched via its console script."""
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
GUI_SOURCE = REPOSITORY_ROOT / "draft" / "gui"

for path in (REPOSITORY_ROOT, GUI_SOURCE):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
