import streamlit as st
import os
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical RAG Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .stChatMessage { border-radius: 12px; margin-bottom: 0.5rem; }
    .source-box {
        background: #f0f4ff;
        border-left: 4px solid #4a6cf7;
        padding: 0.6rem 1rem;
        border-radius: 6px;
        font-size: 0.82rem;
        margin-top: 0.4rem;
    }
    .status-ok  { color: #16a34a; font-weight: 600; }
    .status-err { color: #dc2626; font-weight: 600; }
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

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM_PROMPT = """You are an expert medical assistant trained on clinical guidelines covering:
• Pulmonology & Critical Care  
• Neurology  
• Endocrinology  
• Nephrology & Gastroenterology  
• Infectious Diseases, Haematology & Oncology  
• Rheumatology & Emergency Medicine  

Use ONLY the provided context to answer questions. Be precise, cite drug doses accurately, and flag any safety-critical information. If the answer is not in the context, say so clearly. Never speculate beyond the provided guidelines."""

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in {
    "messages": [],
    "vectorstore": None,
    "docs_loaded": False,
    "groq_client": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helper functions ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(pdf_files):
    """Load PDFs, chunk text, build FAISS index."""
    embeddings = get_embeddings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    all_docs = []

    progress = st.progress(0, text="Processing PDFs…")
    for i, pdf_file in enumerate(pdf_files):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        chunks = splitter.split_documents(pages)
        # tag source
        for chunk in chunks:
            chunk.metadata["source"] = pdf_file.name
        all_docs.extend(chunks)
        os.unlink(tmp_path)
        progress.progress((i + 1) / len(pdf_files), text=f"Processed: {pdf_file.name}")

    progress.empty()
    vectorstore = FAISS.from_documents(all_docs, embeddings)
    return vectorstore, len(all_docs)


def retrieve_context(query: str, k: int = 5):
    """Return top-k relevant chunks + source list."""
    if st.session_state.vectorstore is None:
        return "", []
    results = st.session_state.vectorstore.similarity_search(query, k=k)
    context = "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source','unknown')} | Page {d.metadata.get('page','')+1}]\n{d.page_content}"
        for d in results
    )
    sources = list({
        f"{d.metadata.get('source','?')} (p.{d.metadata.get('page',0)+1})"
        for d in results
    })
    return context, sources


def chat_with_groq(messages_history, model: str):
    """Stream response from Groq."""
    client = st.session_state.groq_client
    return client.chat.completions.create(
        model=model,
        messages=messages_history,
        max_tokens=1024,
        temperature=0.2,
        stream=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    # API Key
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

    # Model selector
    selected_model = st.selectbox("Model", GROQ_MODELS, index=0)
    st.caption("All models above are currently active on Groq.")

    st.divider()

    # PDF Upload
    st.header("📄 Upload Medical PDFs")
    uploaded_pdfs = st.file_uploader(
        "Upload PDF guidelines",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload the clinical guideline PDFs provided.",
    )

    if uploaded_pdfs:
        if st.button("🔄 Build Knowledge Base", use_container_width=True):
            with st.spinner("Building vector store… (first run may take ~60 s)"):
                try:
                    vs, n_chunks = build_vectorstore(uploaded_pdfs)
                    st.session_state.vectorstore = vs
                    st.session_state.docs_loaded = True
                    st.success(f"✅ Indexed {n_chunks} chunks from {len(uploaded_pdfs)} PDFs")
                except Exception as e:
                    st.error(f"Error building index: {e}")

    if st.session_state.docs_loaded:
        st.markdown('<span class="status-ok">📚 Knowledge base ready</span>', unsafe_allow_html=True)

    st.divider()

    # Settings
    st.header("🔧 Retrieval Settings")
    top_k = st.slider("Chunks to retrieve (k)", 3, 10, 5)
    show_sources = st.toggle("Show source references", value=True)

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Built with LangChain · FAISS · HuggingFace · Groq")


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🏥 Medical RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Evidence-based clinical guidelines · Powered by Groq LLM</div>',
    unsafe_allow_html=True,
)

# Status bar
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("API Status", "✅ Ready" if st.session_state.groq_client else "❌ No Key")
with col2:
    st.metric("Knowledge Base", "✅ Loaded" if st.session_state.docs_loaded else "⚠️ Empty")
with col3:
    st.metric("Model", selected_model.split("-")[0].upper())

st.divider()

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if show_sources and msg.get("sources"):
            with st.expander("📎 Sources"):
                for src in msg["sources"]:
                    st.markdown(f'<div class="source-box">📄 {src}</div>', unsafe_allow_html=True)

# Input
if prompt := st.chat_input("Ask a clinical question…"):
    # Guard checks
    if not st.session_state.groq_client:
        st.warning("⚠️ Please enter your Groq API key in the sidebar.")
        st.stop()
    if not st.session_state.docs_loaded:
        st.warning("⚠️ Please upload PDFs and build the knowledge base first.")
        st.stop()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve context
    context, sources = retrieve_context(prompt, k=top_k)

    # Build message list for Groq
    groq_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Include last 6 turns for memory
    for m in st.session_state.messages[-6:]:
        groq_messages.append({"role": m["role"], "content": m["content"]})
    # Inject context into last user turn
    groq_messages[-1]["content"] = (
        f"Context from medical guidelines:\n{context}\n\n"
        f"Question: {prompt}\n\n"
        "Answer based strictly on the context above."
    )

    # Stream response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        try:
            stream = chat_with_groq(groq_messages, selected_model)
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full_response += delta
                placeholder.markdown(full_response + "▌")
                time.sleep(0.005)
            placeholder.markdown(full_response)

            if show_sources and sources:
                with st.expander("📎 Sources"):
                    for src in sources:
                        st.markdown(f'<div class="source-box">📄 {src}</div>', unsafe_allow_html=True)

        except Exception as e:
            full_response = f"❌ Error: {e}"
            placeholder.error(full_response)

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources,
    })
