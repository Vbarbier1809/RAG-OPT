from dotenv import load_dotenv; load_dotenv()
import hashlib, json
from pathlib import Path
import numpy as np
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CACHE = PROC / "embed_cache.npz"
MODEL, DIM, BATCH = "gemini-embedding-001", 768, 50

client = genai.Client(http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(
    attempts=8, initial_delay=5.0, http_status_codes=[429, 503])))

def embed_text(c):
    """Texte envoyé au modèle : on préfixe titre + section pour donner du contexte au chunk."""
    return f"{c['title']} — {c['heading']}\n\n{c['text']}"

chunks = [json.loads(l) for l in (PROC / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
texts = [embed_text(c) for c in chunks]
keys = [hashlib.sha1(f"{MODEL}|{DIM}|{t}".encode()).hexdigest() for t in texts]

# cache : si le script est interrompu (quota...), on reprend là où on s'est arrêté
cache = dict(np.load(CACHE)) if CACHE.exists() else {}
todo = [i for i, k in enumerate(keys) if k not in cache]
print(f"{len(chunks)} chunks | {len(chunks) - len(todo)} déjà en cache | {len(todo)} à calculer")

for b in range(0, len(todo), BATCH):
    idx = todo[b:b + BATCH]
    res = client.models.embed_content(
        model=MODEL, contents=[texts[i] for i in idx],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=DIM),
    )
    for i, e in zip(idx, res.embeddings):
        cache[keys[i]] = np.array(e.values, dtype=np.float32)
    np.savez(CACHE, **cache)
    print(f"  {min(b + BATCH, len(todo))}/{len(todo)}")

V = np.stack([cache[k] for k in keys])
V /= np.linalg.norm(V, axis=1, keepdims=True)   # normaliser : obligatoire quand on tronque sous 3072
np.save(PROC / "embeddings.npy", V)
print(f"embeddings.npy : {V.shape}")
