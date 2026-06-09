# RAG App — FAISS + Gemini on Streamlit

## What this app does
- **Tab 1 – Manage Documents**: Add text documents that get embedded with `all-MiniLM-L6-v2` and stored in a FAISS index.
- **Tab 2 – Semantic Search**: Query the FAISS index directly (no LLM needed).
- **Tab 3 – RAG Chat**: Retrieves relevant docs from FAISS, then sends them as context to Gemini for a grounded answer.

---

## Local setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Set your Gemini key before running:
```bash
export GOOGLE_API_KEY="your-key-here"
```

---

## Deploy on Streamlit Community Cloud

1. Push this folder to a **GitHub repo**.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick your repo.
3. Set the main file path to `app.py`.
4. Under **Advanced settings → Secrets**, add:

```toml
GOOGLE_API_KEY = "your-key-here"
```

> **Note**: Streamlit Cloud injects secrets as environment variables automatically when you use `os.environ.get("GOOGLE_API_KEY")`, which this app already does.

---

## Deploy on other platforms (Railway, Render, HuggingFace Spaces)

Set the `GOOGLE_API_KEY` environment variable in your platform's dashboard and run:

```
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

---

## File persistence note
`faiss_index.bin` and `documents.pkl` are saved to the **working directory**. On stateless platforms (Streamlit Cloud, Render free tier) these files reset on each redeploy. For persistent storage, swap in a cloud solution like AWS S3 or Google Cloud Storage.
