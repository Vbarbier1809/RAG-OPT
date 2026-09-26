from pathlib import Path
import time
import pymupdf, pymupdf4llm

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

for pdf in sorted(RAW.glob("*.pdf")):
    t0 = time.perf_counter()
    doc = pymupdf.open(pdf)
    raw = "\n\n".join(page.get_text() for page in doc)   # méthode 1 : texte brut
    t1 = time.perf_counter()
    md = pymupdf4llm.to_markdown(str(pdf))               # méthode 2 : markdown structuré
    t2 = time.perf_counter()

    (OUT / f"{pdf.stem}.raw.txt").write_text(raw, encoding="utf-8")
    (OUT / f"{pdf.stem}.md").write_text(md, encoding="utf-8")
    print(f"{pdf.name[:40]:40} | {len(doc):3} p. | brut {len(raw):7} car. ({t1-t0:.1f}s) "
          f"| md {len(md):7} car. ({t2-t1:.1f}s)")