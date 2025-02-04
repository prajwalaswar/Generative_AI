# 📄 Gemma Model Document Q&A

A Streamlit-based Document Q&A application using LangChain, FAISS, Google Generative AI Embeddings, and Groq LLM to enable question-answering on PDF documents.

## 🚀 Features
- Load and process PDF documents.
- Embed documents using FAISS and Google Generative AI Embeddings.
- Query documents using a Groq-powered LLM.
- Retrieve contextually relevant document chunks.
- User-friendly Streamlit interface.

## 🛠️ Installation

1. **Clone the repository**
   ```sh
   git clone https://github.com/yourusername/gemma-document-qa.git
   cd gemma-document-qa
   ```

2. **Create a virtual environment (optional but recommended)**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

## 🔑 API Keys Setup
Ensure you have your API keys set up in a `.env` file:

```
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
```

## 📂 Document Storage
Place your PDF documents inside a directory named `us_census/` at the root of the project.

## 🚀 Running the App

Run the Streamlit application with:
```sh
streamlit run app.py
```

## 📝 How It Works
1. Click **"Documents Embedding"** to load and embed PDF files into a FAISS vector database.
2. Enter your question in the text box and hit Enter.
3. The app will retrieve relevant document chunks and answer your query.
4. View document similarity search results for context.

## 🏗️ Technologies Used
- **Streamlit** - Web framework for building UI.
- **LangChain** - Framework for building LLM-powered applications.
- **FAISS** - Efficient similarity search and vector storage.
- **Google Generative AI Embeddings** - Document embedding for vector storage.
- **Groq LLM** - Large Language Model for answering queries.

## 📌 Future Enhancements
- Support for multiple document formats.
- Advanced query processing and summarization.
- UI improvements and additional interactive elements.

## 🤝 Contributing
Feel free to fork this repository, submit issues, and contribute!

## 📜 License
This project is licensed under the MIT License.

## 🙌 Acknowledgments
- Inspired by various open-source LangChain and FAISS projects.
- Powered by Google AI and Groq LLM.

---
**Author**: [Prajwal Aswar](https://github.com/prajwalaswar)  
**GitHub Repo**: [Your Repository Link](https://github.com/prajwalaswar/Generative_AI/new/main/QnA_App)
