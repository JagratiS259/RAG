"""
Healthcare RAG — Clinical Decision Support
Streamlit Web App | Powered by Groq + LangChain + FAISS
Run locally: streamlit run app.py
"""

import os
import streamlit as st
from pathlib import Path

# ── LangChain + Groq imports ───────────────────────────────────────────────
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader,
    UnstructuredHTMLLoader, UnstructuredMarkdownLoader,
    CSVLoader
)
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR        = Path("data")
VECTORSTORE_DIR = Path("vectorstore")
LLM_MODEL       = "llama3-8b-8192"   # Groq model
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"  # Free local embeddings
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 64
TOP_K           = 5

DATA_DIR.mkdir(exist_ok=True)
VECTORSTORE_DIR.mkdir(exist_ok=True)

# ── Clinical Prompt ────────────────────────────────────────────────────────
CLINICAL_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert clinical decision support assistant.
Answer ONLY from the provided medical context below.
If the answer is not in the context, say: "I cannot find sufficient clinical evidence in the provided documents for this question."
Never fabricate drug dosages, diagnoses, or treatment protocols.
Structure your answer with clear bullet points where appropriate.

Context:
{context}

Clinical Question:
{question}

Evidence-Based Clinical Answer:"""
)

# ── Helper functions ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
    )

def load_documents():
    docs = []
    loaders = {
        "*.pdf":  lambda p: PyPDFLoader(str(p)).load(),
        "*.txt":  lambda p: TextLoader(str(p), encoding="utf-8").load(),
        "*.docx": lambda p: Docx2txtLoader(str(p)).load(),
        "*.html": lambda p: UnstructuredHTMLLoader(str(p)).load(),
        "*.htm":  lambda p: UnstructuredHTMLLoader(str(p)).load(),
        "*.md":   lambda p: UnstructuredMarkdownLoader(str(p)).load(),
        "*.csv":  lambda p: CSVLoader(str(p)).load(),
    }
    for pattern, loader_fn in loaders.items():
        for p in DATA_DIR.glob(f"**/{pattern}"):
            try:
                docs.extend(loader_fn(p))
            except Exception as e:
                st.warning(f"⚠️ Could not load {p.name}: {e}")
    # Handle xlsx separately with pandas
    for p in DATA_DIR.glob("**/*.xlsx"):
        try:
            df = pd.read_excel(p)
            text = df.to_string(index=False)
            from langchain_core.documents import Document
            docs.append(Document(page_content=text, metadata={"source": str(p)}))
        except Exception as e:
            st.warning(f"⚠️ Could not load {p.name}: {e}")
    return docs

def split_documents(docs):
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    ).split_documents(docs)

@st.cache_resource(show_spinner=False)
def load_vectorstore_cached():
    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )

@st.cache_resource(show_spinner=False)
def get_llm(api_key: str):
    return ChatGroq(
        model=LLM_MODEL,
        groq_api_key=api_key,
        temperature=0,
    )

def build_rag_chain(vs, llm):
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["query"])),
            question=lambda x: x["query"],
        )
        | CLINICAL_PROMPT
        | llm
        | StrOutputParser()
    )
    return retriever, chain

# ── Page Setup ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical Decision Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #f05537 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    color: white;
    margin-bottom: 1.5rem;
}
.main-header h1 { margin: 0; font-size: 1.8rem; font-weight: 600; }
.main-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.95rem; }
.answer-box {
    background: #eaf4fb;
    border-left: 5px solid #f05537;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    font-size: 0.97rem;
    line-height: 1.75;
    color: #1a1a2e;
    white-space: pre-wrap;
}
.source-card {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.84rem;
    color: #444;
}
.source-title { font-weight: 600; color: #f05537; }
.disclaimer {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    border-radius: 6px;
    padding: 0.65rem 1rem;
    font-size: 0.84rem;
    color: #856404;
    margin-bottom: 1rem;
}
.stat-card {
    background: #f0f7ff;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    text-align: center;
    font-size: 0.85rem;
    color: #1a1a2e;
}
.stat-card span { font-size: 1.3rem; font-weight: 700; display: block; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    api_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        placeholder="Paste your Groq API key here",
        help="Get your free key at https://console.groq.com/keys",
    )

    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", "")

    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
        st.success("✅ API Key set")
    else:
        st.warning("⚠️ Enter your Groq API key to begin")

    st.divider()

    # Model selector
    groq_model = st.selectbox(
        "🤖 Groq Model",
        ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"],
        index=0,
    )
    LLM_MODEL = groq_model

    st.divider()
    st.markdown("## 📂 Document Index")

    uploaded_files = st.file_uploader(
        "Upload Medical Documents",
        type=["pdf", "txt", "docx", "xlsx", "csv", "html", "htm", "md"],
        accept_multiple_files=True,
        help="Upload clinical guidelines, drug references, or protocols",
    )

    if uploaded_files:
        for uf in uploaded_files:
            (DATA_DIR / uf.name).write_bytes(uf.read())
        st.success(f"Uploaded {len(uploaded_files)} file(s)")

    all_files = list(DATA_DIR.glob("**/*.pdf")) + list(DATA_DIR.glob("**/*.txt"))
    if all_files:
        st.markdown(f"**{len(all_files)} file(s) in library:**")
        for f in all_files:
            st.markdown(f"• {f.name}")

    if st.button("🔄 Build / Rebuild Index", type="primary", use_container_width=True):
        with st.spinner("⏳ Indexing documents..."):
            docs   = load_documents()
            chunks = split_documents(docs)
            vs = FAISS.from_documents(chunks, get_embeddings())
            vs.save_local(str(VECTORSTORE_DIR))
            load_vectorstore_cached.clear()
            st.success(f"✅ Indexed {len(chunks)} chunks from {len(docs)} pages!")

    st.divider()
    st.markdown("""
    <div style='font-size:0.8rem; color:#666;'>
    <strong>Models</strong><br>
    LLM: Groq (LLaMA3 / Mixtral)<br>
    Embeddings: HuggingFace MiniLM<br>
    Vector DB: FAISS (local)
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style='font-size:0.78rem; color:#888;'>
    ⚠️ <strong>Disclaimer:</strong> For decision support only.
    Validate all outputs with a licensed clinician.
    </div>
    """, unsafe_allow_html=True)

# ── Main Panel ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
  <h1>🩺 Clinical Decision Support</h1>
  <p>Evidence-based answers grounded in your medical document library · Powered by Groq</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='disclaimer'>
⚠️ <strong>For professional use only.</strong> This tool provides decision support based on uploaded documents.
All clinical decisions must be validated by qualified healthcare professionals.
</div>
""", unsafe_allow_html=True)

EXAMPLES = [
    "What is the first-line treatment for community-acquired pneumonia?",
    "What are the diagnostic criteria for sepsis and the Hour-1 Bundle?",
    "What are absolute contraindications for IV tPA in ischemic stroke?",
    "What is the recommended HbA1c target and medication for Type 2 Diabetes?",
    "What is the preferred treatment for H. pylori eradication?",
    "What are the first-line antihypertensive agents for a diabetic patient?",
]

st.subheader("💬 Ask a Clinical Question")

col_ex, col_gap = st.columns([3, 1])
with col_ex:
    selected_example = st.selectbox(
        "Quick examples",
        ["— type your own question below —"] + EXAMPLES,
        label_visibility="collapsed",
    )

question = st.text_area(
    "Clinical Question",
    value="" if selected_example.startswith("—") else selected_example,
    height=90,
    placeholder="e.g. What antibiotic is recommended for community-acquired pneumonia in ICU patients?",
    label_visibility="collapsed",
)

ask_col, clear_col, _ = st.columns([1, 1, 4])
with ask_col:
    ask_btn = st.button("🔍 Ask", type="primary", use_container_width=True)
with clear_col:
    if st.button("🗑️ Clear", use_container_width=True):
        st.rerun()

# ── Process Query ───────────────────────────────────────────────────────────
if ask_btn:
    if not api_key:
        st.error("⚠️ Please enter your Groq API key in the sidebar.")
    elif not question.strip():
        st.warning("Please enter a clinical question.")
    elif not VECTORSTORE_DIR.exists() or not any(VECTORSTORE_DIR.iterdir()):
        st.error("⚠️ No index found! Upload documents and click **Build / Rebuild Index** first.")
    else:
        with st.spinner("🔬 Searching clinical knowledge base with Groq..."):
            try:
                vs               = load_vectorstore_cached()
                llm              = get_llm(api_key)
                retriever, chain = build_rag_chain(vs, llm)
                sources          = retriever.invoke(question)
                answer           = chain.invoke({"query": question})

                st.markdown("### 🩺 Clinical Answer")
                st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                s1, s2, s3 = st.columns(3)
                with s1:
                    st.markdown(f"<div class='stat-card'><span>{len(sources)}</span>Chunks Retrieved</div>", unsafe_allow_html=True)
                with s2:
                    unique_src = len(set(d.metadata.get("source","?") for d in sources))
                    st.markdown(f"<div class='stat-card'><span>{unique_src}</span>Source Files</div>", unsafe_allow_html=True)
                with s3:
                    total_chars = sum(len(d.page_content) for d in sources)
                    st.markdown(f"<div class='stat-card'><span>{total_chars:,}</span>Chars Analyzed</div>", unsafe_allow_html=True)

                st.markdown("<br>**📚 Source Documents Used:**", unsafe_allow_html=True)
                for i, doc in enumerate(sources, 1):
                    src     = Path(doc.metadata.get("source", "unknown")).name
                    page    = doc.metadata.get("page", "N/A")
                    excerpt = doc.page_content[:280].replace("\n", " ")
                    st.markdown(f"""
                    <div class='source-card'>
                        <span class='source-title'>[{i}] {src}</span> — Page {page}<br>
                        <em>{excerpt}…</em>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Make sure your Groq API key is valid and the index is built.")
