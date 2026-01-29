import streamlit as st
import os
import tempfile
import requests

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="AI PDF Assistant", layout="wide")
st.title("📚 AI Assistant for Your PDFs")

# -------------------- CHAT MEMORY --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------- HF API FUNCTION --------------------
def query_hf_api(prompt):
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}"}

    payload = {
        "inputs": prompt,
        "parameters": {"max_length": 512, "temperature": 0.3}
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    result = response.json()

    if isinstance(result, list):
        return result[0]["generated_text"]
    else:
        return "Model error. Try again."

# -------------------- FILE UPLOAD --------------------
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

# -------------------- PROCESS DOCUMENTS --------------------
if st.button("🔄 Process Documents"):
    if uploaded_files:
        with st.spinner("Processing PDFs..."):
            documents = []

            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                loader = PyPDFLoader(tmp_path)
                documents.extend(loader.load())

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=150
            )
            docs = splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            db = FAISS.from_documents(docs, embeddings)
            db.save_local("faiss_index")

            st.success("Documents processed!")

# -------------------- LOAD VECTOR DB --------------------
if os.path.exists("faiss_index"):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    st.divider()
    st.subheader("💬 Chat with your PDFs")

    # Show chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # User input
    if query := st.chat_input("Ask something from your documents..."):
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").write(query)

        with st.spinner("Thinking..."):
            docs = retriever.invoke(query)
            context = "\n\n".join([d.page_content for d in docs])

            prompt = f"""
Answer using only the context below.

Context:
{context}

Question: {query}
"""

            answer = query_hf_api(prompt)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

        with st.expander("📄 Sources"):
            for d in docs:
                st.write(d.metadata.get("source", "Unknown"))

else:
    st.info("Upload and process PDFs first.")
