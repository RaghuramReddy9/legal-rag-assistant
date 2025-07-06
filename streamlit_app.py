import os
import streamlit as st
from rag_setup import process_pdf_to_chunks, save_uploaded_file
from qa_system import load_vectorstore, build_qa_chain, ask_question, rank_chunks_by_keyword

st.set_page_config(page_title="PDF Legal Q&A Bot", layout="centered")

st.title("📄 Upload Legal PDF & Ask Gemini")

uploaded_file = st.file_uploader("Upload a legal PDF", type="pdf")

if uploaded_file:
    file_path = save_uploaded_file(uploaded_file)

    with st.spinner("Processing document..."):
        chunks = process_pdf_to_chunks(file_path)
        vectorstore = chunks  # optionally save to FAISS
        qa_chain = build_qa_chain(vectorstore)
        st.success("✅ PDF processed and embedded!")

    question = st.text_input("Ask your legal question")

    if st.button("Get Answer") and question:
        # Get LLM answer + source docs
        answer, sources = ask_question(qa_chain, question)
        st.success(answer)

        # Debug view: show raw source context
        st.subheader("📄 Raw sources returned by Gemini:")
        for doc in sources:
            st.code(doc.page_content[:200])  # show only first 200 chars

        # Rank chunks using keyword match
        ranked_chunks = rank_chunks_by_keyword(sources, question)

        # Show ranked results
        with st.expander("📌 Ranked Source Document Snippets"):
            if ranked_chunks:
                for key, info in ranked_chunks.items():
                    st.markdown(f"**{key} – Score: {info['score']}**")
                    st.write(info["text"])
            else:
                st.info("No relevant content found in source documents.")
