# 🏥 Medical RAG Chatbot

#A Retrieval-Augmented Generation (RAG) chatbot for evidence-based clinical guidelines, powered by **Groq LLM**, **FAISS** vector search, and **Streamlit**.

---

## 📋 Included Knowledge Base PDFs

| PDF File | Topics |
|---|---|
| `02_pulmonology_critical_care_txt.pdf` | Asthma, COPD, PE, ARDS, CAP, Pleural Effusion |
| `03_neurology_txt.pdf` | Stroke, Epilepsy, Meningitis, Parkinson's |
| `04_endocrinology_txt.pdf` | Diabetes, DKA, Thyroid, Adrenal Disorders |
| `05_nephrology_gastroenterology_txt.pdf` | AKI, CKD, GI Bleeding, IBD, Liver Disease |
| `06_infections_haematology_oncology_txt.pdf` | Sepsis, HIV, TB, Anaemia, Haematology |
| `07_rheumatology_emergency_txt.pdf` | RA, SLE, Gout, Anaphylaxis, Toxicology |

---

## 🚀 Local Setup

### 1. Clone / download the project
```bash
git clone <your-repo-url>
cd rag_chatbot
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key
Edit `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "gsk_your_actual_key_here"
```
Get a free key at: https://console.groq.com

### 5. Run the app
```bash
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this project to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo → set **Main file**: `app.py`
4. Go to **Advanced settings → Secrets** and add:
   ```
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Click **Deploy** — done!

> ⚠️ Do **not** commit `.streamlit/secrets.toml` to GitHub (it's in `.gitignore`)

---

## 🔧 How to Use

1. Enter your Groq API key in the sidebar (or set it in secrets)
2. Upload the PDF guidelines using the sidebar uploader
3. Click **Build Knowledge Base** — wait ~30–60 seconds
4. Start asking clinical questions!

### Example Questions
- *"What is the CURB-65 score and how is it used?"*
- *"How do you manage acute severe asthma?"*
- *"What are the Wells score criteria for PE?"*
- *"What is the first-line treatment for DKA?"*
- *"When should you start NIV in COPD exacerbation?"*

---

## 🤖 Available Groq Models

| Model | Speed | Best For |
|---|---|---|
| `llama-3.3-70b-versatile` | Fast | Best quality (recommended) |
| `llama-3.1-8b-instant` | Fastest | Quick answers |
| `llama3-70b-8192` | Fast | Stable fallback |
| `mixtral-8x7b-32768` | Fast | Long context |
| `gemma2-9b-it` | Fast | Lightweight |

---

## 🏗️ Architecture

```
User Question
     │
     ▼
HuggingFace Embeddings (all-MiniLM-L6-v2)
     │
     ▼
FAISS Vector Search → Top-K relevant chunks
     │
     ▼
Groq LLM (llama-3.3-70b) with context
     │
     ▼
Streamed Answer + Source Citations
```

---

## ⚠️ Disclaimer

This chatbot is for **educational and reference purposes only**. Always verify clinical decisions with current guidelines and consult a qualified healthcare professional.
