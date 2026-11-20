"""Tick checklist items by number and refresh the progress line.

Usage: python scripts/tick.py 1 2 3
"""

import re
import sys
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "PROJECT_CHECKLIST.md"
text = path.read_text()
for n in sys.argv[1:]:
    text = re.sub(rf"^- \[ \] {n}\. ", f"- [x] {n}. ", text, flags=re.M)
done = len(re.findall(r"^- \[x\] \d+\.", text, flags=re.M))
total = len(re.findall(r"^- \[[ x]\] \d+\.", text, flags=re.M))
text = re.sub(r"\*\*Progress: .*?\*\*", f"**Progress: {done} / {total} ({done / total:.0%})**", text)
path.write_text(text)
print(f"{done}/{total} = {done / total:.0%}")
