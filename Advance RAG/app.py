import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
import time
from dotenv import load_dotenv
load_dotenv()

st.title("🚀 Advanced RAG ChatGroq")


groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("Missing GROQ_API_KEY in environment!")
    st.stop()


llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-8b-8192"
)


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

if "vectorstore" not in st.session_state:
    
    web_docs = WebBaseLoader("https://docs.smith.langchain.com/").load()
    
    pdf_docs = PyPDFLoader("sample.pdf").load() if os.path.exists("sample.pdf") else []

    all_docs = web_docs + pdf_docs

    
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

   
    vectorstore = FAISS.from_documents(chunks, embeddings)

    st.session_state.vectorstore = vectorstore
    st.success("✅ Vector DB built with hybrid documents")


vector_retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})

keyword_retriever = st.session_state.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})


retriever = EnsembleRetriever(retrievers=[vector_retriever, keyword_retriever], weights=[0.7, 0.3])


prompt = ChatPromptTemplate.from_template(
    """
    Use the context below to answer the user's question.
    Be accurate, detailed, and avoid guessing.
    
    <context>
    {context}
    </context>

    Question: {input}
    """
)


document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)


user_input = st.text_input("🔎 Ask something about the documents...")

if user_input:
    with st.spinner("Thinking..."):
        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_input})
        duration = time.process_time() - start

  
    st.subheader("💬 Response")
    st.write(response["answer"])
    st.caption(f"⏱️ Response time: {round(duration, 2)} seconds")

    
    with st.expander("📄 Retrieved Chunks"):
        for doc in response["context"]:
            st.write(doc.metadata.get("source", "Unknown Source"))
            st.markdown(doc.page_content)
            st.markdown("---")
