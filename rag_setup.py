import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

def load_and_chunk_pdf(file_path, chunk_size=500, chunk_overlap=50):
    # Load PDF using LangChain PyPDFLoader
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split into chunks for semantic search
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)

    return chunks


def embed_and_save(chunks, index_path="faiss_index/"):
    # Create Embeddings for each chunk
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create FAISS vector index
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save FAISS index to disk
    vectorstore.save_local(index_path)
    print(f"✅ FAISS index saved to: {index_path}")

if __name__ == "__main__":
    file_path = "data/sample_terms.pdf"
    if not os.path.exists(file_path):
        print("⚠️ File not found.")
    else:
        chunks = load_and_chunk_pdf(file_path)
        print(f"✅ Chunks Ready: {len(chunks)}")
        print("🔍 Preview:\n", chunks[0].page_content[:300])

        embed_and_save(chunks) 

