"""Make `src/` importable without relying on an editable install.

(On macOS, files under synced folders can get the `hidden` flag, and Python 3.13+
skips hidden .pth files, which silently breaks `pip install -e`.)
"""

import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
