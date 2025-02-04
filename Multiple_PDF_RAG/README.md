📖 Gemma Model Document Q&A
This project is a Streamlit-based Q&A application that leverages LangChain, FAISS, and Gemini AI embeddings to extract information from PDF documents. Users can upload PDFs, generate vector embeddings, and ask questions based on the document content.

🚀 Features
📄 PDF Document Ingestion: Upload and process PDFs.
🔍 Vector Search with FAISS: Converts documents into embeddings for efficient retrieval.
🤖 AI-Powered Q&A: Uses Llama3-8b-8192 via Groq API to answer queries based on document content.
📌 Contextual Retrieval: Ensures responses are based strictly on the provided context.
⚡ Fast Processing: Efficient text chunking and vector retrieval using Google Generative AI Embeddings.
📦 Installation
First, clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/gemma-document-qa.git
cd gemma-document-qa
Create a virtual environment and install dependencies:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
🔑 API Key Setup
Create a .env file in the project directory and add your API keys:

ini
Copy
Edit
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
🏃‍♂️ Running the App
Start the Streamlit application:

bash
Copy
Edit
streamlit run app.py
🎯 How to Use
Upload PDF Documents: Place your PDFs inside the ./us_census folder.
Generate Vector Embeddings: Click the Documents Embedding button to process the files.
Ask Questions: Type a question in the text box and get AI-generated responses.
View Source Context: Expand the Document Similarity Search section to check relevant document chunks.
🛠️ Future Improvements
✅ Support for multiple document formats (e.g., Word, TXT).
✅ Interactive file upload feature in Streamlit.
✅ Enhanced multi-turn conversation support.
✨ Credits
Developed by: Prajwal Aswar
