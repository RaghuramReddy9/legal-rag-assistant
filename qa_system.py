import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load Gemini API key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Load FAISS vector store
def load_vectorstore():
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
    return vectorstore

#  Create QA chain from retriever
def build_qa_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0,)
    retriever = vectorstore.as_retriever()

    prompt = PromptTemplate.from_template("""
You are a legal assistant. Use the following context to answer.

Context:
{context}

Question:
{question}

Answer:
""")

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

# Ask a question
def ask_question(chain, question):
    result = chain.invoke({"query": question})
    return result["result"], result["source_documents"]
    
# Main loop for testing
if __name__ == "__main__":
    vs = load_vectorstore()
    qa_chain = build_qa_chain(vs)

    while True:
        question = input("\nAsk your legal question (or 'exit'): ")
        if question.lower() == "exit":
            break

        answer, sources = ask_question(qa_chain, question)
        print("\n Answer:", answer)

        print("\n Source context:")
        for doc in sources:
            print("-", doc.page_content[:100])