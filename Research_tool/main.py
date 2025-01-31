import os
import streamlit as st
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

# Streamlit app setup
st.set_page_config(
    page_title="RockyBot: Stock  Tool 📈",
    page_icon="📊",
    layout="wide",
)
st.title("📰 RockyBot: News Research Tool")
st.markdown(
    """
    Welcome to **RockyBot**! This tool helps you analyze and research news articles quickly and efficiently.
    Upload URLs of articles, ask questions, and get insights in seconds! 🚀
    """
)

# Sidebar setup
st.sidebar.title("🔗 Input URLs")
st.sidebar.markdown("Paste up to 3 article URLs for analysis:")

# Sidebar input for URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}:", placeholder=f"Enter URL {i+1} here")
    urls.append(url)

process_url_clicked = st.sidebar.button("📥 Process URLs")

# FAISS vector store file path
faiss_index_path = "faiss_index"

# Main content placeholder
main_placeholder = st.empty()

# Initialize the Groq-based LLM
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-8b-8192",
)

# Function to process URLs and create FAISS vector store
def process_urls(urls):
    with st.spinner("🚀 Loading data from URLs..."):
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()

    with st.spinner("🔄 Splitting data into chunks..."):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000,
        )
        docs = text_splitter.split_documents(data)

    with st.spinner("⚙️ Creating FAISS vector store..."):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore_google = FAISS.from_documents(docs, embeddings)

        # Save FAISS index locally
        vectorstore_google.save_local(faiss_index_path)

    st.success("✅ FAISS index saved successfully!")

# Process URLs when the button is clicked
if process_url_clicked:
    if all(urls):
        process_urls(urls)
    else:
        st.error("Please provide all the URLs before processing.")

# User query input
st.markdown("---")
query = st.text_input("🔍 Enter your question:", placeholder="Type your question here...")

# Handle user query
if query:
    if os.path.exists(faiss_index_path):
        with st.spinner("🤖 Fetching the answer..."):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectorstore = FAISS.load_local(
                faiss_index_path,
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

        st.markdown("## 🧠 Answer:")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.markdown("### 🔗 Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.markdown(f"- {source}")
    else:
        st.error("FAISS index not found. Please process the URLs first.")

# Footer
st.markdown("---")
st.markdown(
    """
    **👨‍💻 Created by [Prajwal Aswar](https://github.com/prajwalaswar?tab=repositories)**  
    Powered by [LangChain](https://github.com/langchain-ai) and [Streamlit](https://streamlit.io/). 🚀
    """
)
