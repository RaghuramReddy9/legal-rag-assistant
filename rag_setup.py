import os
from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Loads .env file
load_dotenv()  

# Set your key from env
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Load legal document
loader = TextLoader("data/terms_and_conditions.txt")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Create Gemini embeddings
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Store in FAISS
vectorstore = FAISS.from_documents(chunks, embedding)
vectorstore.save_local("faiss_index")

print("✅ Vector index created and saved with Gemini embeddings.")
