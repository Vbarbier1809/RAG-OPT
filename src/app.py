from pathlib import Path
import streamlit as st

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"

st.set_page_config(page_title="RAG lab", page_icon="📚", layout="wide")
st.title("📚 RAG lab")

if not (PROC / "embeddings.npy").exists():
    st.warning("Pas encore d'embeddings. Lance d'abord dans le terminal :\n\n"
               "```\npython src/chunk.py\npython src/embed.py\n```\n\npuis recharge la page.")
    st.stop()

import rag   # chargé seulement si l'index existe (rag.py lit embeddings.npy à l'import)

with st.sidebar:
    st.header("Réglages")
    k = st.slider("Nombre de chunks récupérés (top-k)", 1, 15, rag.TOP_K)
    st.caption(f"{len(rag.chunks)} chunks · {len({c['doc_id'] for c in rag.chunks})} documents")
    st.caption(f"Embeddings : {rag.EMBED_MODEL} ({rag.DIM} dim.)  \nGénération : {rag.GEN_MODEL}")
    if st.button("Effacer la conversation"):
        st.session_state.history = []

def show_sources(hits):
    with st.expander(f"Sources ({len(hits)})"):
        for n, h in enumerate(hits, start=1):
            st.markdown(f"**[{n}]** `{h['score']:.2f}` · **{h['title']}** — {h['heading']} · "
                        f"p. {h['page']} · [arXiv {h['doc_id']}](https://arxiv.org/abs/{h['doc_id']})")
            st.text(h["text"][:1500])

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("hits"):
            show_sources(msg["hits"])

if question := st.chat_input("Pose une question sur tes documents…"):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Recherche et génération…"):
            try:
                answer, hits = rag.ask(question, k)
            except Exception as e:
                answer, hits = f"❌ Erreur : {e}", []
            rag.langfuse.flush()
        st.markdown(answer)
        if hits:
            show_sources(hits)
    st.session_state.history.append({"role": "assistant", "content": answer, "hits": hits})
