import streamlit as st
import os
import time
import tempfile
import re
import hashlib
import numpy as np
from groq import Groq
from pypdf import PdfReader
import faiss

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

CHUNK_SIZE    = 600
CHUNK_OVERLAP = 100
VOCAB_SIZE    = 8000   # TF-IDF vocabulary size

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "messages":    [],
    "chunks":      [],
    "index":       None,
    "vocab":       None,
    "idf":         None,
    "docs_loaded": False,
    "groq_client": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Pure-numpy TF-IDF embeddings (zero external deps) ────────────────────────
def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    # simple bigrams for better medical term matching
    unigrams = tokens
    bigrams  = [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    return unigrams + bigrams


def build_vocab(all_texts: list[str]) -> tuple[dict, np.ndarray]:
    """Build vocabulary and IDF from corpus."""
    from collections import Counter
    tf_counts = []
    df_counter = Counter()

    for text in all_texts:
        tokens = set(tokenize(text))
        df_counter.update(tokens)
        tf_counts.append(tokens)

    # pick top VOCAB_SIZE tokens by document frequency
    top_tokens = [t for t, _ in df_counter.most_common(VOCAB_SIZE)]
    vocab = {tok: i for i, tok in enumerate(top_tokens)}

    N = len(all_texts)
    idf = np.zeros(len(vocab), dtype=np.float32)
    for tok, idx in vocab.items():
        df = df_counter.get(tok, 0)
        idf[idx] = np.log((N + 1) / (df + 1)) + 1.0   # smoothed IDF

    return vocab, idf


def tfidf_vector(text: str, vocab: dict, idf: np.ndarray) -> np.ndarray:
    from collections import Counter
    tokens = tokenize(text)
    tf = Counter(tokens)
    total = max(len(tokens), 1)
    vec = np.zeros(len(vocab), dtype=np.float32)
    for tok, cnt in tf.items():
        if tok in vocab:
            idx = vocab[tok]
            vec[idx] = (cnt / total) * idf[idx]
    norm = np.linalg.norm(vec) + 1e-9
    return vec / norm


def embed_texts(texts: list[str], vocab: dict, idf: np.ndarray) -> np.ndarray:
    return np.vstack([tfidf_vector(t, vocab, idf) for t in texts])


# ── PDF parsing ───────────────────────────────────────────────────────────────
def parse_pdf(file) -> list[dict]:
    reader = PdfReader(file)
    chunks = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        start = 0
        while start < len(text):
            chunk_text = text[start:start + CHUNK_SIZE].strip()
            if len(chunk_text) > 50:
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
        progress.progress((i + 0.5) / len(pdf_files), text=f"Parsing: {f.name}")
        all_chunks.extend(parse_pdf(f))

    progress.progress(0.7, text="Building vocabulary…")
    texts = [c["text"] for c in all_chunks]
    vocab, idf = build_vocab(texts)

    progress.progress(0.85, text="Vectorising chunks…")
    vecs = embed_texts(texts, vocab, idf)

    progress.progress(0.95, text="Building FAISS index…")
    dim   = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    progress.progress(1.0, text="Done!")
    progress.empty()
    return all_chunks, index, vocab, idf


# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(query: str, k: int = 5):
    vec = tfidf_vector(query, st.session_state.vocab, st.session_state.idf)
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
                chunks, index, vocab, idf = build_index(uploaded)
                st.session_state.chunks      = chunks
                st.session_state.index       = index
                st.session_state.vocab       = vocab
                st.session_state.idf         = idf
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

    st.caption("Embeddings: TF-IDF (local) · Vector DB: FAISS · LLM: Groq")


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🏥 Medical RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Evidence-based clinical guidelines · Powered by Groq</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.metric("API",   "✅ Ready"  if st.session_state.groq_client else "❌ No Key")
c2.metric("KB",    "✅ Loaded" if st.session_state.docs_loaded else "⚠️ Empty")
c3.metric("Model", selected_model.split("-")[0].upper())

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
