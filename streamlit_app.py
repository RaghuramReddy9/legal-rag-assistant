import streamlit as st
from qa_system import load_vectorstore, build_qa_chain, ask_question

st.set_page_config(page_title="Legal Q&A Bot", page_icon="⚖️")

st.title("⚖️ Gemini-Powered Legal Assistant")
st.markdown("Ask any question based on your uploaded legal terms.")

# Load vectorstore + QA chain
with st.spinner("Loading AI brain..."):
    vectorstore = load_vectorstore()
    qa_chain = build_qa_chain(vectorstore)

# Input from user
question = st.text_input("📝 Ask your legal question:")

if st.button("Get Answer"):
    if question:
        with st.spinner("Thinking..."):
            answer, sources = ask_question(qa_chain, question)
            st.success(answer)

            with st.expander("📄 Source Context"):
                for doc in sources:
                    st.write(doc.page_content)
    else:
        st.warning("Please enter a question.")
