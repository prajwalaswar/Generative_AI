# 🚀 Advanced RAG with LangChain, Groq, and FAISS

## 📌 Overview
This project implements an advanced **Retrieval-Augmented Generation (RAG)** pipeline using **LangChain**, **Groq LLM (Mixtral-8x7B)**, **FAISS**, and **Google Generative AI Embeddings**. The system retrieves contextually relevant documents from a web source and uses an LLM to generate accurate responses based on the retrieved data.

## 🛠️ Features
- **Web Scraping & Data Ingestion**: Loads documents from a web source dynamically.
- **FAISS-based Vector Search**: Stores and retrieves document chunks efficiently.
- **Google AI Embeddings**: Converts text into vector embeddings for semantic search.
- **Mixtral-8x7B (Groq) LLM**: Generates accurate responses based on retrieved content.
- **Context-Aware Prompting**: Ensures the model only answers based on the retrieved documents.
- **Performance Monitoring**: Measures response time for optimization.

## 📦 Tech Stack
- **LangChain**: Manages the retrieval and LLM pipeline.
- **Streamlit**: Provides an interactive UI for user input and response display.
- **Groq (Mixtral-8x7B)**: Processes and generates responses.
- **FAISS**: Handles vector storage and retrieval.
- **Google Generative AI**: Generates embeddings for document search.

## 🚀 Installation
### 1️⃣ Clone the Repository:
```bash
git clone https://github.com/your-username/advanced-rag-langchain.git
cd advanced-rag-langchain
```

### 2️⃣ Install Dependencies:
```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up API Keys:
Create a `.env` file and add your API keys:
```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
```

### 4️⃣ Run the Application:
```bash
streamlit run app.py
```

## 📌 How It Works
1. Loads documents from the specified URL.
2. Splits them into smaller chunks with overlapping tokens.
3. Generates embeddings using Google Generative AI.
4. Stores embeddings in a FAISS vector database.
5. Retrieves relevant chunks using semantic search.
6. Passes the retrieved context to the Groq LLM for response generation.
7. Displays the answer along with document sources for transparency.

## 🎯 Usage
- Enter a question in the Streamlit UI.
- The system retrieves relevant documents and generates an answer.
- Expand the "Document Similarity Search" section to see the retrieved chunks.

## 📌 Future Improvements
- 🔹 **Hybrid Retrieval** (BM25 + FAISS) for improved accuracy.
- 🔹 **Metadata Filtering** (filter documents by date or topic).
- 🔹 **Multi-Document Support** (ingest and search across multiple sources).

## 📝 License
This project is licensed under the **MIT License**.

---
👨‍💻 Developed by Prajwal Aswar

