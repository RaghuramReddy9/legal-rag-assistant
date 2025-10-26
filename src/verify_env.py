import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from google import generativeai as genai

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Google Gemini API connection 
def test_gemini_connection():
    print(" Testing Gemini 2.0 Flash connection...")
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        model = client.models.get("gemini-2.0-flash-001")
        print(f"Connected to model: {model.name}")
    except Exception as e:
        print(" Gemini connection failed:", e)


# FAISS + Embeddings 
def test_vectorstore(index_path="data/legal_index"):
    print("\n Testing FAISS vectorstore loading...")
    try:
        embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(index_path, embedder, allow_dangerous_deserialization=True)
        print(f" Loaded FAISS index from: {index_path}")
        return vectorstore
    except Exception as e:
        print(" Failed to load FAISS index:", e)
        return None


# RAG pipeline test 
def test_rag_pipeline(vectorstore):
    print("\nRunning sample RAG query...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    from src.llm import get_llm
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template("""
    You are a legal assistant. 
    Use the provided context to answer the question.
    If unsure, say "I’m not sure based on the document."

    Context:
    {context}

    Question: {question}
    """)

    rag_chain = (
        RunnableMap({
            "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            "question": RunnablePassthrough(),
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    query = "What is the refund policy period?"
    print(f" Query: {query}")
    try:
        answer = rag_chain.invoke(query)
        print("\n RAG system working! Example Answer:\n", answer[:400])
    except Exception as e:
        print("RAG pipeline failed:", e)


if __name__ == "__main__":
    print("Running full environment verification...\n")
    test_gemini_connection()
    vs = test_vectorstore()
    if vs:
        test_rag_pipeline(vs)
    print("\nEnvironment test complete.")
