import streamlit as st
import os
import time
import json
import tempfile
import numpy as np
from groq import Groq
from pypdf import PdfReader
import faiss
import requests

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical RAG Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header{font-size:2rem;font-weight:700;color:#1a1a2e;text-align:center;margin-bottom:.25rem}
.sub-header{font-size:1rem;color:#555;text-align:center;margin-bottom:1.5rem}
.source-box{background:#f0f4ff;border-left:4px solid #4a6cf7;padding:.6rem 1rem;
            border-radius:6px;font-size:.82rem;margin-top:.4rem}
.status-ok{color:#16a34a;font-weight:600}
.status-err{color:#dc2626;font-weight:600}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

SYSTEM_PROMPT = """You are an expert medical assistant trained on clinical guidelines covering:
Pulmonology & Critical Care, Neurology, Endocrinology, Nephrology & Gastroenterology,
Infectious Diseases, Haematology & Oncology, Rheumatology & Emergency Medicine.

Use ONLY the provided context to answer. Be precise, cite drug doses accurately, and flag
safety-critical information. If the answer is not in the context, say so clearly."""

CHUNK_SIZE   = 600   # characters per chunk
CHUNK_OVERLAP = 100

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "messages":    [],
    "chunks":      [],      # list of {"text":…, "source":…, "page":…}
    "index":       None,    # faiss index
    "docs_loaded": False,
    "groq_client": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Embedding via HuggingFace Inference API (no torch needed) ─────────────────
HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

@st.cache_data(show_spinner=False)
def get_embeddings_hf(texts: tuple) -> np.ndarray:
    """Call HuggingFace free Inference API — no local torch required."""
    headers = {"Content-Type": "application/json"}
    payload = {"inputs": list(texts), "options": {"wait_for_model": True}}
    resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # API returns list[list[float]]
    return np.array(data, dtype=np.float32)


def embed_single(text: str) -> np.ndarray:
    return get_embeddings_hf((text,))[0]


def embed_batch(texts: list[str], batch_size=32) -> np.ndarray:
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        vecs = get_embeddings_hf(tuple(batch))
        all_vecs.append(vecs)
    return np.vstack(all_vecs)


# ── PDF parsing ───────────────────────────────────────────────────────────────
def parse_pdf(file) -> list[dict]:
    reader = PdfReader(file)
    chunks = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        # sliding window chunking
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end].strip()
            if len(chunk_text) > 50:          # skip tiny fragments
                chunks.append({
                    "text":   chunk_text,
                    "source": file.name,
                    "page":   page_num + 1,
                })
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── Build FAISS index ─────────────────────────────────────────────────────────
def build_index(pdf_files):
    all_chunks = []
    progress = st.progress(0, text="Parsing PDFs…")

    for i, f in enumerate(pdf_files):
        progress.progress((i + 0.5) / len(pdf_files), text=f"Parsing {f.name}…")
        all_chunks.extend(parse_pdf(f))

    progress.progress(0.8, text="Generating embeddings (HF API)…")
    texts = [c["text"] for c in all_chunks]
    vecs  = embed_batch(texts)

    # L2-normalise for cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    vecs  = vecs / norms

    dim   = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product = cosine after normalisation
    index.add(vecs)

    progress.progress(1.0, text="Done!")
    progress.empty()
    return all_chunks, index


# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(query: str, k: int = 5):
    vec = embed_single(query)
    vec = vec / (np.linalg.norm(vec) + 1e-9)
    vec = vec.reshape(1, -1)
    _, ids = st.session_state.index.search(vec, k)
    results = [st.session_state.chunks[i] for i in ids[0] if i >= 0]
    context = "\n\n---\n\n".join(
        f"[{r['source']} | p.{r['page']}]\n{r['text']}" for r in results
    )
    sources = list({f"{r['source']} (p.{r['page']})" for r in results})
    return context, sources


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_…")
    if api_key:
        try:
            st.session_state.groq_client = Groq(api_key=api_key)
            st.markdown('<span class="status-ok">✅ API Key set</span>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<span class="status-err">❌ {e}</span>', unsafe_allow_html=True)
    else:
        st.info("Enter your Groq API key to begin.")

    st.divider()
    selected_model = st.selectbox("Model", GROQ_MODELS)

    st.divider()
    st.header("📄 Upload Medical PDFs")
    uploaded = st.file_uploader(
        "Upload PDF guidelines", type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded:
        if st.button("🔄 Build Knowledge Base", use_container_width=True):
            try:
                chunks, index = build_index(uploaded)
                st.session_state.chunks      = chunks
                st.session_state.index       = index
                st.session_state.docs_loaded = True
                st.success(f"✅ Indexed {len(chunks)} chunks from {len(uploaded)} PDFs")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.docs_loaded:
        st.markdown('<span class="status-ok">📚 Knowledge base ready</span>', unsafe_allow_html=True)

    st.divider()
    top_k        = st.slider("Chunks to retrieve (k)", 3, 10, 5)
    show_sources = st.toggle("Show source references", value=True)

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Embeddings: HF Inference API · Vector DB: FAISS · LLM: Groq")


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🏥 Medical RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Evidence-based clinical guidelines · Powered by Groq</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.metric("API",    "✅ Ready"  if st.session_state.groq_client else "❌ No Key")
c2.metric("KB",     "✅ Loaded" if st.session_state.docs_loaded else "⚠️ Empty")
c3.metric("Model",  selected_model.split("-")[0].upper())

st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if show_sources and msg.get("sources"):
            with st.expander("📎 Sources"):
                for s in msg["sources"]:
                    st.markdown(f'<div class="source-box">📄 {s}</div>',
                                unsafe_allow_html=True)

if prompt := st.chat_input("Ask a clinical question…"):
    if not st.session_state.groq_client:
        st.warning("⚠️ Enter your Groq API key in the sidebar.")
        st.stop()
    if not st.session_state.docs_loaded:
        st.warning("⚠️ Upload PDFs and build the knowledge base first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    context, sources = retrieve(prompt, k=top_k)

    groq_msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in st.session_state.messages[-6:]:
        groq_msgs.append({"role": m["role"], "content": m["content"]})
    groq_msgs[-1]["content"] = (
        f"Context from medical guidelines:\n{context}\n\n"
        f"Question: {prompt}\n\nAnswer strictly from the context above."
    )

    with st.chat_message("assistant"):
        placeholder   = st.empty()
        full_response = ""
        try:
            stream = st.session_state.groq_client.chat.completions.create(
                model=selected_model, messages=groq_msgs,
                max_tokens=1024, temperature=0.2, stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full_response += delta
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)

            if show_sources and sources:
                with st.expander("📎 Sources"):
                    for s in sources:
                        st.markdown(f'<div class="source-box">📄 {s}</div>',
                                    unsafe_allow_html=True)
        except Exception as e:
            full_response = f"❌ Error: {e}"
            placeholder.error(full_response)

    st.session_state.messages.append({
        "role": "assistant", "content": full_response, "sources": sources,
    })
