# ⚖️Gemini Legal Q&A Bot (RAG Pipeline)

This project sets up a **Retrieval-Augmented Generation (RAG)** pipeline that can **answer legal questions** based on a user-uploaded legal document.

---

##  Tech Stack

- 🧠 **Google Gemini 1.5 Flash** – for embeddings + answers  
- 🦜 **LangChain** – for LLM orchestration  
- 📁 **FAISS** – for local vector database  
- 🖼️ **Streamlit** – for interactive frontend (optional)

---

##  How It Works

1. `rag_setup.py` – Loads and chunks legal `.txt` file → creates embeddings → stores in FAISS  
2. `qa_system.py` – Accepts questions, retrieves relevant chunks, gets Gemini-powered answer  
3. `streamlit_app.py` – (Optional) Clean UI for querying your doc like ChatGPT  

---

## 📁 Project Structure

legal-rag-bot/
├── data/ # Holds legal .txt documents
│ └── terms_and_conditions.txt
├── faiss_index/ # Saved vector DB
├── rag_setup.py # One-time setup script
├── qa_system.py # Terminal-based QA script
├── streamlit_app.py # UI app (optional)
├── requirements.txt
└── README.md

---

## ⚙️ Setup & Run

### 1. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate   # or source .venv/bin/activate on macOS/Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add a legal file
Put your `.txt` legal file (T&C, contract, etc.) in the `data/` folder.

### 4. Set your API key

Add your Gemini API key in a `.env` file:
```bash
GOOGLE_API_KEY=your-real-api-key
```
### 5. Build the vector index
```bash
python rag_setup.py
```
### 6. Ask questions (terminal)
```bash
python qa_system.py
```
### 7. Or launch the UI
```bash
streamlit run streamlit_app.py
```
##  Example Question
```bash
What are the refund terms mentioned in the agreement?
```
🔁 The bot finds and returns the answer with citations from your legal text.

## Use Cases: 

    1. Customer policy search

    2. HR document understanding

    3. Contract clause extraction

    4. Compliance assistant

    5. Internal knowledge agents

## Note:

Keep your `.env` file private. Do not commit API keys. Use `.gitignore` to keep secrets secure ✅

##  Author

**Raghuramreddy Thirumalareddy**

- 🔗 [GitHub](https://github.com/RaghuramReddy9)
- 💼 [LinkedIn](https://www.linkedin.com/in/raghuramreddy-ai)





