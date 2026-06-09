# 🩺 Healthcare RAG — Clinical Decision Support
### Google Gemini + LangChain + FAISS + Streamlit

---

## 📁 Project Structure

```
healthcare_rag_gemini/
├── Healthcare_RAG_Gemini.ipynb   ← Jupyter notebook (full walkthrough)
├── app.py                         ← Streamlit web application
├── requirements.txt               ← Python dependencies
├── data/
│   └── clinical_guidelines.txt   ← Sample clinical data (add your PDFs here)
├── vectorstore/                   ← FAISS index (auto-created after indexing)
└── .streamlit/
    └── secrets.toml               ← API key config for Streamlit Cloud
```

---

## ⚡ Quick Start (Local)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Gemini API key
```bash
export GOOGLE_API_KEY="your-api-key-here"
```
Get your key at: https://aistudio.google.com/app/apikey (free tier available)

### 3. Run the Streamlit app
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

### 4. In the app:
1. Enter your Gemini API key in the sidebar (or it reads from env)
2. Upload your medical PDFs (optional — sample data is pre-loaded)
3. Click **Build / Rebuild Index**
4. Ask clinical questions!

---

## ☁️ Deploy on Streamlit Cloud (Free)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Healthcare RAG with Gemini"
git remote add origin https://github.com/YOUR_USERNAME/healthcare-rag.git
git push -u origin main
```
> ⚠️ Add `vectorstore/` and `.streamlit/secrets.toml` to `.gitignore`

### Step 2 — Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io
2. Click **New app**
3. Select your GitHub repo
4. Set **Main file path**: `app.py`
5. Click **Deploy**

### Step 3 — Add your Gemini API Key as a Secret
1. In your deployed app, click **⋮ → Settings → Secrets**
2. Add:
```toml
GOOGLE_API_KEY = "your-gemini-api-key-here"
```
3. Click **Save** — the app restarts automatically

> Note: On Streamlit Cloud, FAISS index must be rebuilt each session (or connect to a persistent store like Pinecone for production).

---

## 🔑 Getting a Gemini API Key (Free)

1. Visit: https://aistudio.google.com/app/apikey
2. Sign in with your Google account
3. Click **Create API key**
4. Copy and use in the app

**Free tier limits:** 15 RPM, 1M tokens/min, 1500 requests/day — plenty for development!

---

## 🧠 Architecture

```
Medical PDFs / TXT files
         │
         ▼
   Document Loader (LangChain)
         │
         ▼
   Text Splitter — 512 chars, 64 overlap
         │
         ▼
   Google embedding-001  ──►  FAISS Index (saved locally)
                                    │
                          ┌─────────┘
                          │  Similarity Search (Top-5)
                          ▼
                  Clinical Prompt Template
                          │
                          ▼
                  Gemini 1.5 Flash (temp=0)
                          │
                          ▼
              Answer + Source Documents
```

---

## ⚙️ Configuration (in app.py)

| Parameter | Default | Description |
|---|---|---|
| `LLM_MODEL` | `gemini-1.5-flash` | Switch to `gemini-1.5-pro` for higher quality |
| `EMBED_MODEL` | `models/embedding-001` | Google's embedding model |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K` | `5` | Chunks retrieved per query |

---

## ⚠️ Disclaimer

This system is a **clinical decision support tool only**.  
All outputs must be reviewed and validated by a licensed healthcare professional before clinical application.
