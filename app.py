import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
import pickle

# ─── File paths for persistence ───────────────────────────────────────────────
FAISS_INDEX_FILE = "faiss_index.bin"
DOCUMENTS_FILE   = "documents.pkl"

# ─── Load resources (cached) ──────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    model     = SentenceTransformer("all-MiniLM-L6-v2")
    dimension = 384
    index     = faiss.IndexFlatL2(dimension)
    documents = []

    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(DOCUMENTS_FILE):
        try:
            index = faiss.read_index(FAISS_INDEX_FILE)
            with open(DOCUMENTS_FILE, "rb") as f:
                documents = pickle.load(f)
            st.toast("✅ Loaded existing FAISS index & documents.", icon="💾")
        except Exception as e:
            st.warning(f"Could not load persisted data: {e}. Starting fresh.")
    return model, index, documents

model, index, initial_documents = load_resources()

# ─── Session state ─────────────────────────────────────────────────────────────
if "documents"    not in st.session_state:
    st.session_state.documents    = initial_documents
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─── Gemini setup ──────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
gemini_model   = None

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")

def generate_llm_response(query_text: str) -> str:
    if gemini_model is None:
        return (
            "⚠️ Gemini not configured. "
            "Set the `GOOGLE_API_KEY` environment variable and restart the app."
        )
    try:
        response = gemini_model.generate_content(query_text)
        return response.text
    except Exception as e:
        return f"Error generating response: {e}"

# ─── RAG helper: retrieve context + ask Gemini ────────────────────────────────
def rag_answer(query: str, top_k: int = 3) -> str:
    """Retrieve relevant docs from FAISS, then ask Gemini with that context."""
    if index.ntotal == 0:
        return generate_llm_response(query)   # fallback: pure LLM

    q_vec = model.encode([query]).astype("float32")
    k_val = min(top_k, index.ntotal)
    _, idxs = index.search(q_vec, k_val)

    context_docs = [
        st.session_state.documents[i]
        for i in idxs[0] if i != -1
    ]
    context = "\n\n".join(context_docs)

    prompt = (
        f"You are a helpful assistant. Use the context below to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    return generate_llm_response(prompt)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG – FAISS + Gemini", layout="wide", page_icon="🔍")
st.title("🔍 RAG: FAISS Semantic Search + Gemini Chat")
st.caption("Index your documents, search them semantically, or chat with Gemini grounded in your data.")

tab1, tab2, tab3 = st.tabs(["📄 Manage Documents", "🔎 Semantic Search", "💬 RAG Chat"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Document management
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.subheader("Add a Document")
        new_doc = st.text_area(
            "Paste text, a paragraph, FAQ, policy snippet…",
            height=160,
            placeholder="Type or paste your content here.",
            key="new_doc_input",
        )

        if st.button("📥 Index Document", use_container_width=True):
            if new_doc.strip():
                vector = model.encode([new_doc]).astype("float32")
                index.add(vector)
                st.session_state.documents.append(new_doc.strip())
                st.success(f"Indexed! Total documents: **{index.ntotal}**")
            else:
                st.warning("Enter some text first.")

    with col2:
        st.subheader("Persistence")
        if st.button("💾 Save Index", use_container_width=True):
            try:
                faiss.write_index(index, FAISS_INDEX_FILE)
                with open(DOCUMENTS_FILE, "wb") as f:
                    pickle.dump(st.session_state.documents, f)
                st.success("Saved successfully.")
            except Exception as e:
                st.error(f"Save failed: {e}")

        if st.button("🗑️ Clear All", use_container_width=True):
            dimension = 384
            # reassign through cache workaround
            faiss.write_index(faiss.IndexFlatL2(dimension), FAISS_INDEX_FILE)
            index.__init__(dimension)   # reset in-memory index
            st.session_state.documents = []
            for f in [FAISS_INDEX_FILE, DOCUMENTS_FILE]:
                if os.path.exists(f):
                    os.remove(f)
            st.success("Cleared. Please reload the page.")

    st.divider()
    st.subheader(f"Indexed Library — {index.ntotal} document(s)")
    if st.session_state.documents:
        for i, doc in enumerate(st.session_state.documents):
            with st.expander(f"Document #{i + 1} — {doc[:60]}…" if len(doc) > 60 else f"Document #{i + 1}"):
                st.write(doc)
    else:
        st.info("No documents indexed yet. Add one above.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Pure semantic search (no LLM)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Search Your Document Library")
    query_s = st.text_input("Search query:", placeholder="Enter a question or phrase…", key="sem_query")
    top_k_s = st.slider("Results (K):", 1, 10, 3, key="sem_k")

    if st.button("🚀 Search", use_container_width=True, key="sem_btn"):
        if not query_s.strip():
            st.warning("Enter a query.")
        elif index.ntotal == 0:
            st.error("No documents indexed yet.")
        else:
            q_vec  = model.encode([query_s]).astype("float32")
            k_val  = min(top_k_s, index.ntotal)
            dists, idxs = index.search(q_vec, k_val)

            st.subheader("Results")
            for rank, (dist, idx) in enumerate(zip(dists[0], idxs[0]), start=1):
                if idx != -1:
                    with st.expander(f"Rank #{rank} | Distance: {dist:.4f}", expanded=True):
                        st.write(st.session_state.documents[idx])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – RAG Chat (FAISS retrieval + Gemini generation)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Chat Grounded in Your Documents")

    if not GOOGLE_API_KEY:
        st.warning(
            "Set the `GOOGLE_API_KEY` environment variable to enable this tab. "
            "Pure semantic search in **Tab 2** works without an API key."
        )

    # Render history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Ask something…"):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer = rag_answer(user_input)
                st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
