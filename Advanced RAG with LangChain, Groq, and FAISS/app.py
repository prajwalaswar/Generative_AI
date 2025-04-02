import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

## Load the Groq API key
groq_api_key = os.environ["GROQ_API_KEY"]

# Initialize Ollama embeddings and load documents if not already in session state
if "vector" not in st.session_state:
    # Initialize embeddings with base_url
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load documents from the provided web URL
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()

    # Split documents into smaller chunks
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])

    # Create FAISS vector store
    st.session_state.vectors = FAISS.from_documents(
        st.session_state.final_documents,
        st.session_state.embeddings
    )

# Streamlit app title
st.title("ChatGroq Demo")

# Initialize ChatGroq model
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="mixtral-8x7b-32768"
)

# Create a prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Set up document chain and retriever
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# User input in Streamlit
prompt = st.text_input("Input your prompt here:")

if prompt:
    # Measure response time
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    st.write("Response time:", time.process_time() - start)

    # Display the response
    st.write(response['answer'])

    # Display the document similarity search in an expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
