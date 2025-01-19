# ChatGroq Demo with LangChain

This repository contains a Streamlit-based demonstration of **ChatGroq**, integrating **LangChain** for document retrieval and question answering. The app uses **Groq API**, **Google Generative AI Embeddings**, and a **FAISS vector store** for efficient document similarity search and contextual responses.

---

## 🚀 Features
- **Document Retrieval**: Uses FAISS vector store for quick and efficient document search.
- **Embeddings**: Powered by Google Generative AI Embeddings (`embedding-001`).
- **Chat Model**: ChatGroq model (`mixtral-8x7b-32768`) provides accurate and context-aware answers.
- **Web Document Loading**: Dynamically loads and processes documents from a web URL.
- **Interactive Interface**: Built with Streamlit for real-time Q&A and document similarity exploration.

---

## 📦 Dependencies
Install the required libraries with:
```bash
pip install streamlit langchain faiss-cpu langchain-groq langchain-community langchain-google-genai python-dotenv
