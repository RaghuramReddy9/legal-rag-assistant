# ⚖️ AI Legal Assistant (Gemini + RAG + FAISS)

This project sets up a **Retrieval-Augmented Generation (RAG)** pipeline that can **answer legal questions** based on a user-uploaded legal document.

- Gemini Pro for LLM reasoning
- HuggingFace Embeddings (`all-MiniLM-L6-v2`)
- FAISS for vector search
- LangChain for chaining + context retrieval
- Streamlit UI
---

##  Features

- Upload any legal PDF (Terms, Policies, Contracts)
- Documents are chunked, embedded, and indexed locally
- Ask legal questions in natural language
- Gemini answers with context + cited chunks

---

## Tech Stack

| Layer       | Tool |
|-------------|------|
| Embeddings  | HuggingFace (`sentence-transformers`) |
| Vector DB   | FAISS |
| LLM Answer  | Gemini Pro (via `google-generativeai`) |
| Framework   | LangChain |
| Interface   | Streamlit |


---

## ⚙️ Setup & Run

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/legal-rag-bot
cd legal-rag-bot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate   # or source .venv/bin/activate on macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Add your Gemini API key
Create a .env file:
```bash
GOOGLE_API_KEY=your-gemini-api-key
```
### 5. Run the app
```bash
streamlit run app.py
```


🔁 The bot finds and returns the answer with citations from your legal text.

## Real-World Use Case
This project simulates a legal assistant that helps users:
```bash
 1. Understand refund policies

 2. Clarify cancellation terms

 3. Interpret contract clauses

Ideal for fintech, legaltech, SaaS onboarding, or customer service AI.
```
## Note:

Keep your `.env` file private. Do not commit API keys. Use `.gitignore` to keep secrets secure ✅

##  Author

**Raghuramreddy Thirumalareddy**

- 🔗 [GitHub](https://github.com/RaghuramReddy9)
- 💼 [LinkedIn](https://www.linkedin.com/in/raghuramreddy-ai)

## 📎 License
```bash
MIT — Free to use and adapt.
```




