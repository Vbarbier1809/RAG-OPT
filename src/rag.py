from dotenv import load_dotenv; load_dotenv()
import json, sys
from pathlib import Path
import numpy as np
from google import genai
from google.genai import types
from langfuse import get_client, observe
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
EMBED_MODEL, GEN_MODEL, DIM, TOP_K = "gemini-embedding-001", "gemini-flash-latest", 768, 5

PROMPT = """Tu es un assistant qui répond à des questions sur un corpus d'articles scientifiques.
Réponds uniquement à partir des extraits ci-dessous. Cite tes sources avec leur numéro, ex. [1], [3].
Si les extraits ne permettent pas de répondre, dis-le clairement.
Réponds dans la langue de la question.

EXTRAITS :
{context}

QUESTION : {question}"""

langfuse = get_client()
GoogleGenAIInstrumentor().instrument()   # trace automatiquement les appels Gemini
client = genai.Client(http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(
    attempts=5, initial_delay=2.0, http_status_codes=[429, 503])))

chunks = [json.loads(l) for l in (PROC / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
V = np.load(PROC / "embeddings.npy")

@observe(name="retrieve")
def retrieve(question: str, k: int = TOP_K):
    res = client.models.embed_content(
        model=EMBED_MODEL, contents=[question],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=DIM),
    )
    q = np.array(res.embeddings[0].values)
    q /= np.linalg.norm(q)
    scores = V @ q                                   # similarité cosinus (vecteurs normalisés)
    best = np.argsort(-scores)[:k]
    return [{**chunks[i], "score": float(scores[i])} for i in best]

@observe(name="rag")
def ask(question: str, k: int = TOP_K):
    hits = retrieve(question, k)
    context = "\n\n".join(
        f"[{n}] ({h['title']} — {h['heading']}, p. {h['page']})\n{h['text']}"
        for n, h in enumerate(hits, start=1))
    rep = client.models.generate_content(model=GEN_MODEL, contents=PROMPT.format(context=context, question=question))
    return rep.text, hits

def show(question):
    answer, hits = ask(question)
    print(f"\n{answer}\n\nSources :")
    for n, h in enumerate(hits, start=1):
        print(f"  [{n}] {h['score']:.2f} | {h['doc_id']} p.{h['page']} | {h['title'][:50]} — {h['heading'][:40]}")

if __name__ == "__main__":   # lancé en terminal (pas quand app.py l'importe)
    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]))
    else:
        print(f"{len(chunks)} chunks chargés. Pose ta question (Entrée vide pour quitter).")
        while question := input("\n> ").strip():
            show(question)
    langfuse.flush()  # indispensable dans un script court, sinon la trace peut ne jamais partir
