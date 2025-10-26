import os 
import asyncio
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from llm import get_llm

def load_vectorstore(index_path="data/faiss_index"):
    """Load the saved FAISS vector store containing document embeddings."""
    print("Loading FAISS vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

def build_streaming_rag_chain(vectorstore):
    """Modern RAG pipeline with streaming support.
    Streams Gemini 2.0 Flash responses in real time."""
    print("Building modern RAG chain...")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()

    # Define the dynamic prompt
    template = """
    You are a helpful legal AI assistant.
    Use the provided context to answer precisely.
    If unsure, say "I am not certain from this document."

    Context:
    {context}

    Question: {question}

    Provide the answer in a natural and professional tone.
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Define RAG chain
    def join_docs(docs):
        """Combine retriever docs into one text block for LLM"""
        return "\n\n---\n\n".join([doc.page_content for doc in docs])                                                  

    # Streaming RAG chain
    rag_chain = (
        RunnableMap(
            {
                "context": retriever | join_docs,
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def get_sources(vectorstore, query, top_k=3):
    """
    Retrieve top-k relevant document chunks for the given query.
    Returns a list of snippet strings.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(query)
    sources = [d.page_content[:300] + "..." for d in docs]
    return sources

async def stream_answer(rag_chain, query):
    """
    Streams tokens from Gemini as they are generated.
    Useful for live display in CLI or Streamlit.
    """
    print(f"Streaming answer for: {query}\n")
    async for chunk in rag_chain.astream(query):
        print(chunk, end="", flush=True)
    print("\n\nStreaming complete.")

def run_streaming_test():
    """CLI test for streaming responses."""
    vectorstore = load_vectorstore("data/faiss_index")
    rag_chain = build_streaming_rag_chain(vectorstore)
    query = "what this document about?, and summarize it"
    asyncio.run(stream_answer(rag_chain, query))


if __name__ == "__main__":
    run_streaming_test()

