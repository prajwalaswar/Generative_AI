# Import necessary libraries and modules
import streamlit as st  # Streamlit is used to build the web app's interactive user interface.
import os  # Provides a way to interact with the operating system (e.g., manage environment variables).
from langchain_groq import ChatGroq  # Used to interact with Groq-based language models.
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Splits large text into smaller chunks for processing.
from langchain.chains.combine_documents import create_stuff_documents_chain  # Combines document-based outputs into a coherent response.
from langchain_core.prompts import ChatPromptTemplate  # Creates structured prompts for LLM interactions.
from langchain.chains import create_retrieval_chain  # Creates a retrieval-based pipeline for document search.
from langchain_community.vectorstores import FAISS  # FAISS is a vector store for efficient similarity search.
from langchain_community.document_loaders import PyPDFDirectoryLoader  # Loads PDF documents from a directory.
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Creates embeddings using Google GenAI.
from dotenv import load_dotenv  # Loads environment variables from a .env file.
import os  # Redundant but included to set environment variables.

# Load environment variables from .env
load_dotenv()

# Retrieve API keys for Groq and Google services
groq_api_key = os.getenv('GROQ_API_KEY')  # Get Groq API key from environment variables.
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")  # Set Google API key as an environment variable.

# Streamlit: Set up the application title
st.title("Bhagavad Gita Chatbot \n How can I serve you today ?")  # Displays the app title in the user interface.

# Initialize the Groq-based language model
llm = ChatGroq(
    groq_api_key=groq_api_key,  # API key for authenticating with Groq.
    model_name="Llama3-8b-8192"  # Specifies the Groq model to use.
)

# Define the prompt template for Groq
prompt = ChatPromptTemplate.from_template(
    """
    You are a knowledgeable assistant based on the teachings of the Bhagavad Gita.
    Whenever a user asks a question, provide an answer by referring to the Bhagavad Gita context.
    Make sure to include the relevant shloka (if applicable) and explain it clearly in a way that relates to the user's question.
    Your response should be insightful, compassionate, and deeply rooted in the principles of the Bhagavad Gita.

    <context>
    {context}
    <context>
    
    User's Question: {input}
    
    Your Answer (include relevant shloka and its explanation):
    """
)


# Function to generate embeddings and build a vector store
def vector_embedding():
    # Check if embeddings and vectors are already in session state
    if "vectors" not in st.session_state:
        # Create embeddings using Google GenAI
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # Load PDF documents from the specified directory
        st.session_state.loader = PyPDFDirectoryLoader("./bg")  # Load files from the './us_census' directory.
        st.session_state.docs = st.session_state.loader.load()  # Load the content of the documents.
        # Split the text into smaller chunks for better processing
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # Split the first 20 documents into manageable chunks
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        # Create a FAISS vector store for similarity search using the embeddings
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Streamlit input box for user questions
prompt1 = st.text_input("Plz type here your problem I will provide you the solution According to Bhagavad gita")  # User inputs their question related to the documents.

# Button to trigger the embedding process
if st.button("Documents Embedding"):
    vector_embedding()  # Calls the function to build the vector store.
    st.write("Vector Store DB Is Ready")  # Notifies the user that embeddings are ready.

# Import time to calculate response time
import time

# Process the user's question if input is provided
if prompt1:
    # Create a document processing chain using the language model and prompt
    document_chain = create_stuff_documents_chain(llm, prompt)
    # Retrieve the document retriever from the vector store
    retriever = st.session_state.vectors.as_retriever()
    # Create a retrieval chain that links the retriever and document chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Measure the response time
    start = time.process_time()  # Start time tracking.
    response = retrieval_chain.invoke({'input': prompt1})  # Pass the user's question into the retrieval chain.
    print("Response time :", time.process_time() - start)  # Print the time taken for the response.
    st.write(response['answer'])  # Display the answer from the retrieval chain.

    # Expandable section to show relevant document content
    with st.expander("Document Similarity Search"):
        # Loop through the context chunks and display them
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)  # Show the document's content.
            st.write("--------------------------------")  # Separator for better readability.
