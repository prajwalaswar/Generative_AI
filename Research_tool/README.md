# RockyBot: Stock Tool 📈

## Overview
RockyBot is a news research tool that allows users to analyze and extract insights from online news articles. The tool leverages FAISS vector stores and LLM-based retrieval to provide concise answers to user queries based on article content.

## Features
- 📰 Fetch news articles from URLs
- 🔍 Process and store information in FAISS vector store
- 🤖 Answer questions using Llama3-8b-8192 model via Groq API
- 🔗 Provide source references for answers
- 🛠️ Powered by LangChain, FAISS, and Google Generative AI Embeddings

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rockybot.git
   cd rockybot
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   ```bash
   export GROQ_API_KEY="your_groq_api_key"
   ```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

## How It Works
1. Enter up to 3 article URLs in the sidebar.
2. Click the **Process URLs** button to extract and store content in a FAISS vector store.
3. Enter your question in the input box and get an answer based on the processed articles.

## Dependencies
- Python 3.8+
- Streamlit
- LangChain
- FAISS
- Google Generative AI Embeddings
- Unstructured
- dotenv

## Author
👨‍💻 **Prajwal Aswar**  
🔗 [GitHub](https://github.com/prajwalaswar?tab=repositories)

## License
This project is licensed under the MIT License.

