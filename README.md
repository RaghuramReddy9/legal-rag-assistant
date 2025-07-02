#  Gemini Legal Q&A Bot (RAG Pipeline)

This project sets up a simple Retrieval-Augmented Generation (RAG) system using:
-  LangChain for orchestration
-  Gemini 1.5 Flash for embedding + answering
-  FAISS for fast local vector database

##  How It Works
1. `rag_setup.py`: Loads a legal document, chunks it, embeds with Gemini, stores in FAISS
2. `qa_system.py`: Lets you ask questions and get context-grounded answers

##  Project Structure
legal-rag-bot/
├── data/
│ └── terms_and_conditions.txt
├── faiss_index/
├── rag_setup.py
├── requirements.txt
└── README.md

## 🚀 To Run

1. Create `.venv` and activate it:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:

    `pip install -r requirements.txt`

3. Add your legal `.txt` file to `data/`

4. Set your Gemini API key inside `rag_setup.py`

5. Run:

    `python rag_setup.py`

