import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama

st.set_page_config(page_title="Free AI PDF Assistant", layout="wide")
st.title("📚 Free AI Assistant for Your PDFs")

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if st.button("🔄 Process Documents"):
    if uploaded_files:
        documents = []

        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            documents.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        docs = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(docs, embeddings)
        db.save_local("faiss_index")

        st.success("Documents processed!")

if os.path.exists("faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = ChatOllama(model="llama3")

    st.divider()

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if query := st.chat_input("Ask something..."):
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").write(query)

        docs = retriever.invoke(query)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
Answer using only this context:

{context}

Question: {query}
"""

        response = llm.invoke(prompt)
        answer = response.content

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

        with st.expander("📄 Sources"):
            for d in docs:
                st.write(d.metadata.get("source", "Unknown"))
else:
    st.info("Upload and process PDFs first.")
