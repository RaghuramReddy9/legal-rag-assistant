import os
import streamlit as st
from rag_setup import load_and_chunk_pdf, embed_and_save
from qa_system import load_vectorstore, build_qa_chain, ask_question

st.set_page_config(page_title="Legal RAG Bot", layout="wide")
st.title("📄 AI Legal Assistant (Gemini + RAG + FAISS)")

# -------- File Upload ----------
st.sidebar.subheader("📄 Upload Your Legal PDF")
uploaded_file = st.sidebar.file_uploader("Upload a legal terms PDF", type="pdf")

if uploaded_file:
    with st.spinner("🔄 Processing PDF..."):
        save_path = os.path.join("data", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        chunks = load_and_chunk_pdf(save_path)
        embed_and_save(chunks)
        st.session_state.qa_chain = build_qa_chain(load_vectorstore())
        st.success("✅ PDF indexed and ready for questions!")

# -------- QA Panel -------------
st.sidebar.header("🔍 Legal Search")
question = st.text_input("Ask your legal question:")

if "qa_chain" not in st.session_state:
    try:
        st.session_state.qa_chain = build_qa_chain(load_vectorstore())
    except:
        st.warning("📂 Upload a PDF first to build the chatbot.")
        st.stop()

if st.button("Submit") and question:
    with st.spinner("🤔 Thinking..."):
        answer, sources = ask_question(st.session_state.qa_chain, question)
        st.markdown("### 💡 Answer")
        st.success(answer)

        if sources:
            st.markdown("### 📄 Source Context")
            for i, doc in enumerate(sources):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content[:500])
