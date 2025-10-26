import os 
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# config 
DATA_DIR = "data"
PDF_PATH = os.path.join(DATA_DIR, "software-license-agreement.pdf")  # any document
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")

def build_vector_index(pdf_path=PDF_PATH, index_path=INDEX_PATH):
    """
    1. load a pdf
    2. split it into
    3. convert each chunk into embeddings
    4. save a FAISS index to disk
    """
    print(f"Loading PDF from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print("Splitting document into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    print("Generating embeddings using MiniLM model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Building FAISS vector store...")
    vectorestore = FAISS.from_documents(chunks, embeddings)

    print(f"Saving FAISS index to {index_path}...")
    vectorestore.save_local(index_path)

    print("Vector store built and saved successfully.")
    return vectorestore

if __name__ == "__main__":
    build_vector_index()


