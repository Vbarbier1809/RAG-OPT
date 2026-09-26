import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLEAN, OUT = ROOT / "data" / "clean", ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

TARGET, MAX_PARA, MIN_CHUNK, OVERLAP = 1200, 2000, 100, 400   # en caractères
HEADING = re.compile(r"^#{1,6}\s+(.*)$")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

def clean_title(line):
    return re.sub(r"[*_#]", "", line).strip()

def paragraphs(text):
    """Découpe en paragraphes ; les paragraphes trop longs sont redécoupés par phrases."""
    for p in re.split(r"\n\s*\n", text):
        p = p.strip()
        if len(p) <= MAX_PARA:
            if p:
                yield p
            continue
        buf = ""
        for s in SENTENCE_END.split(p):
            if buf and len(buf) + len(s) > TARGET:
                yield buf
                buf = ""
            buf = f"{buf} {s}".strip()
        if buf:
            yield buf

def chunk_doc(doc):
    segs = [s for s in doc["segments"] if s["section"] != "references"]
    title = next((clean_title(l) for s in segs for l in s["text"].splitlines() if HEADING.match(l)), doc["doc_id"])
    chunks, buf, heading, start = [], [], title, None

    def flush():
        text = "\n\n".join(p for _, p, _ in buf)
        if len(text) >= MIN_CHUNK:
            chunks.append({"doc_id": doc["doc_id"], "title": title, "heading": buf[0][2],
                           "page": buf[0][0], "text": text})

    for seg in segs:
        for p in paragraphs(seg["text"]):
            m = HEADING.match(p.splitlines()[0])
            if m:
                heading = clean_title(m.group(1))
            buf.append((seg["page"], p, heading))
            if sum(len(x[1]) for x in buf) >= TARGET:
                flush()
                last = buf[-1]
                buf = [last] if len(last[1]) < OVERLAP else []   # chevauchement : on garde le dernier paragraphe s'il est court
    if buf:
        flush()
    return chunks

all_chunks = []
for f in sorted(CLEAN.glob("*.json")):
    chunks = chunk_doc(json.loads(f.read_text(encoding="utf-8")))
    all_chunks += chunks
    print(f"{f.stem:14} | {len(chunks):3} chunks | {chunks[0]['title'][:60]}")

for i, c in enumerate(all_chunks):
    c["id"] = i
with open(OUT / "chunks.jsonl", "w", encoding="utf-8") as fh:
    for c in all_chunks:
        fh.write(json.dumps(c, ensure_ascii=False) + "\n")
tailles = sorted(len(c["text"]) for c in all_chunks)
print(f"\n{len(all_chunks)} chunks | taille médiane {tailles[len(tailles) // 2]} car. | max {tailles[-1]} car.")
