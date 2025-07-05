import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import tempfile

# Load Gemini key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="PDF Legal Q&A Bot", page_icon="📄")
st.title("📄 Upload Legal PDF & Ask Gemini")

uploaded_file = st.file_uploader("Upload a legal PDF", type="pdf")

if uploaded_file:
    with st.spinner("Reading your PDF..."):
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Load with LangChain
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # Split
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        # Embed
        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(chunks, embedding)

        # Setup retriever + LLM
        retriever = vectorstore.as_retriever()
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

        prompt = PromptTemplate.from_template("""
You are a legal assistant. Use the context to answer clearly.

Context:
{context}

Question:
{question}

Answer:
""")

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        # Ask questions
        question = st.text_input("📝 Ask your legal question")

        if st.button("Get Answer") and question:
            with st.spinner("Thinking..."):
                result = qa_chain.invoke({"query": question})
                st.success(result["result"])

                with st.expander("📄 Source Document Snippets"):
                    for doc in result["source_documents"]:
                        st.write(doc.page_content[:300])

