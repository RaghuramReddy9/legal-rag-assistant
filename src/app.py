import streamlit as st
from rag_setup import build_vector_index
from qa_system import load_vectorstore, build_streaming_rag_chain, get_sources
import asyncio
import os

st.set_page_config(page_title="AI Legal Assistant", page_icon="⚖️", layout="centered")

# sidebar
st.sidebar.title("📄 Upload Legal Document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

DATA_DIR = "data"
PDF_PATH = os.path.join(DATA_DIR, "uploaded_legal_doc.pdf")

# Upload and process PDF
if uploaded_file is not None:
    # Save uploaded file
    with open(PDF_PATH, "wb") as f:
        f.write(uploaded_file.read())
    st.sidebar.success("✅ PDF uploaded successfully!")

    # Build vector index
    with st.spinner("Building knowledge base..."):
        build_vector_index(pdf_path=PDF_PATH, index_path="data/faiss_index")
        st.sidebar.success("✅ Document processed and indexed!")

# Session state to chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Title and Input
st.title("⚖️ AI Legal Assistant (Gemini 2.0 Flash + RAG)")
st.markdown("Ask questions about your uploaded legal document.")

# chat clearing
if st.button("🧹Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["user"])
    with st.chat_message("assistant"):
        st.markdown(chat["bot"])    

# Chat input
user_query = st.chat_input("💬 Ask a legal question:")

# when user sends message
if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    if not os.path.exists("data/faiss_index"):
        st.error("Please upload and index a PDF first.")
    else:
        vectorstore = load_vectorstore("data/faiss_index")
        rag_chain = build_streaming_rag_chain(vectorstore)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            answer_buffer = []

            async def display_stream():
                async for chunk in rag_chain.astream(user_query):
                    answer_buffer.append(chunk)
                    placeholder.markdown("".join(answer_buffer))

            asyncio.run(display_stream())
            st.success("✅ Response complete!")

            #  Save message in memory 
            full_answer = "".join(answer_buffer)
            st.session_state.chat_history.append({
                "user": user_query,
                "bot": full_answer
            })

            # Sources display
            st.markdown("### 📚 Sources Used:")
            sources = get_sources(vectorstore, user_query)
            for i, src in enumerate(sources, start=1):
                with st.expander(f"Source {i}"):
                    st.write(src)