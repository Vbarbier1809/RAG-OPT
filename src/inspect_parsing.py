import re
from collections import Counter
from pathlib import Path

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"

for f in sorted(PROC.iterdir()):
    txt = f.read_text(encoding="utf-8")
    lines = [l.strip() for l in txt.splitlines() if l.strip() and not l.startswith("|")]
    repetees = [l for l, n in Counter(lines).items() if n >= 4 and len(l) < 80]
    coupures = len(re.findall(r"\w-\n\w", txt))            # mots coupés en fin de ligne
    ligatures = len(re.findall(r"[\ufb00-\ufb06]", txt))    # ﬁ, ﬂ, ﬀ mal décodées
    m = re.search(r"^[#*\s]*\d*\.?\s*references\b", txt, re.I | re.M)
    biblio = f"{100 * (1 - m.start() / len(txt)):.0f}%" if m else "?"
    print(f"{f.name[:22]:22} | coupures {coupures:4} | ligatures {ligatures:3} "
          f"| biblio+annexes {biblio:>4} | répétées : {repetees[:3]}")