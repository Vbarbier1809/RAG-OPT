import json, re
from collections import Counter
from pathlib import Path
import pymupdf4llm

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "clean"
OUT.mkdir(parents=True, exist_ok=True)

PICTURE = re.compile(r"<!-- Start of picture text -->.*?<!-- End of picture text -->", re.S)
PAGE_NUM = re.compile(r"^\s*\d{1,3}\s*$")
MARKERS = [
    (re.compile(r"^[#*\s]*\d*\.?\s*references\b", re.I | re.M), "references"),
    (re.compile(r"^[#*\s]*(appendix|appendices|supplementary)\b", re.I | re.M), "appendix"),
]

def remove_boilerplate(pages):
    """Retire les lignes présentes sur au moins 30 % des pages (en-têtes, pieds de page)."""
    counts = Counter(l for t in pages for l in {x.strip() for x in t.splitlines() if x.strip()})
    seuil = max(3, 0.3 * len(pages))
    parasites = {l for l, n in counts.items() if n >= seuil and l != "```"}
    cleaned = ["\n".join(l for l in t.splitlines()
                         if l.strip() not in parasites and not PAGE_NUM.match(l))
               for t in pages]
    return cleaned, parasites

def split_sections(pages):
    """Découpe chaque page en segments étiquetés body / references / appendix."""
    section, segs = "body", []
    for num, text in enumerate(pages, start=1):
        cuts = sorted((m.start(), sec) for rx, sec in MARKERS for m in rx.finditer(text))
        start = 0
        for pos, new in cuts:
            segs.append({"page": num, "section": section, "text": text[start:pos]})
            start, section = pos, new
        segs.append({"page": num, "section": section, "text": text[start:]})
    return [s for s in segs if s["text"].strip()]

for pdf in sorted(RAW.glob("*.pdf")):
    pages = [PICTURE.sub("", p["text"]) for p in pymupdf4llm.to_markdown(str(pdf), page_chunks=True)]
    pages, parasites = remove_boilerplate(pages)
    segs = split_sections(pages)
    (OUT / f"{pdf.stem}.json").write_text(
        json.dumps({"doc_id": pdf.stem, "segments": segs}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    parts = Counter()
    for s in segs:
        parts[s["section"]] += len(s["text"])
    total = sum(parts.values())
    repart = " ".join(f"{k} {100 * v / total:.0f}%" for k, v in parts.items())
    print(f"{pdf.stem:14} | {len(parasites):2} lignes parasites | {repart}")