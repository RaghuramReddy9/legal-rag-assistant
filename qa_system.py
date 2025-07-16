import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Load FAISS index
def load_vectorstore(index_path="faiss_index/"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

# Create Gemini QA Chain with Custom Prompt
def build_qa_chain(vectorstore):
    prompt = PromptTemplate.from_template("""
You are a helpful legal assistant.
Use the following legal document context to answer the user’s question clearly and concisely.

Context:
{context}

Question:
{question}

Answer:
""")

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    retriever = vectorstore.as_retriever()

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

# Ask a question using the QA chain
def ask_question(chain, question):
    result = chain.invoke({"query": question})
    return result["result"], result["source_documents"]

# Run loop for manual testing
if __name__ == "__main__":
    vs = load_vectorstore()
    qa_chain = build_qa_chain(vs)

    while True:
        query = input("\nAsk a legal question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        answer, sources = ask_question(qa_chain, query)

        print("\n✅ Answer:\n", answer)
        print("\n📄 Source Document(s):")
        for doc in sources:
            print("-", doc.page_content[:300])  # Preview each chunk
