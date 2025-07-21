from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from llm import get_llm

def load_vectorstore():
   embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
   return FAISS.load_local("faiss_index", embedder, allow_dangerous_deserialization=True)

def build_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    llm = get_llm("gemini")  # Gemini for answering
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

def ask_question(chain, query):
    result = chain({"query": query})
    answer = result["result"]
    sources = result.get("source_documents", [])
    return answer, sources
