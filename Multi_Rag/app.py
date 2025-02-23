import streamlit as st
import json
import os
import requests
from dotenv import load_dotenv
from googleapiclient.discovery import build
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
import chromadb
from langchain_groq import ChatGroq

# Load API Keys from .env
load_dotenv(dotenv_path=".env", override=True)

# Get API Keys
youtube_api_key = os.getenv("YOUTUBE_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Check API Key Presence
if not youtube_api_key or not google_api_key or not serper_api_key or not groq_api_key:
    st.error("API Keys are missing! Set them in the .env file.")
    st.stop()

# Initialize YouTube API
youtube = build("youtube", "v3", developerKey=youtube_api_key)

# Initialize Embeddings and ChromaDB
text_embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=google_api_key)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
vector_store = Chroma(client=chroma_client, collection_name="medical_info", embedding_function=text_embedder)

# Initialize Groq API
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-8b-8192"
)

# Function to Retrieve Contextual Information
def get_contextual_info(query):
    results = vector_store.similarity_search(query, k=1)
    return results[0].metadata if results else "No relevant stored information found."

# Function to Search YouTube Videos
def search_youtube_videos(query, max_results=3):
    request = youtube.search().list(
        q=query, part="snippet", type="video", maxResults=max_results
    )
    response = request.execute()
    
    videos = []
    for item in response.get("items", []):
        video_id = item["id"]["videoId"]
        title = item["snippet"]["title"]
        description = item["snippet"]["description"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        videos.append({"title": title, "description": description, "url": video_url})

    return videos

# Function to Search Google (Using Serper.dev)
def google_search(query):
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": serper_api_key, "Content-Type": "application/json"}
    data = {"q": query}

    response = requests.post(url, headers=headers, json=data)
    data = response.json()
    
    return [res["snippet"] for res in data.get("organic", [])[:3]] if "organic" in data else ["No relevant Google search results."]

# Function to Store External Data in ChromaDB
def store_external_data(query):
    google_results = google_search(query)
    youtube_results = search_youtube_videos(query)
    
    combined_info = f"Google Search: {' '.join(google_results)}\nYouTube Insights: {' '.join([video['title'] for video in youtube_results])}"
    
    # Store combined information in ChromaDB
    vector_store.add_texts([combined_info])

# Function to Generate Final Response Using Groq
def generate_final_response(query):
    # Retrieve contextual information from multiple sources
    stored_info = get_contextual_info(query)

    # Extract text content from stored_info if it's a dictionary or list
    if isinstance(stored_info, list) and len(stored_info) > 0:
        stored_info = stored_info[0].page_content  # Extract actual text
    elif isinstance(stored_info, dict):
        stored_info = stored_info.get("text", "No relevant stored information found.")
    else:
        stored_info = "No relevant stored information found."

    google_results = google_search(query)
    youtube_results = search_youtube_videos(query)

    # Combine all results
    combined_info = f"""
    🔹 **Stored Information:** {stored_info}\n
    🔹 **Google Search Results:** {' '.join(google_results)}\n
    🔹 **YouTube Insights:** {' '.join([video['title'] for video in youtube_results])}
    """

    # Generate a refined answer using Groq
    response = llm.invoke(combined_info)

    return response if response else "No relevant answer found."

    
    # Generate a refined answer using Groq
    response = llm.invoke(stored_info)
    return response if response else "No relevant answer found."

# Streamlit UI
st.set_page_config(page_title="Medical AI Search", layout="wide")
st.title("🩺 AI Medical Search & Videos (Powered by Groq)")
st.markdown("Get expert AI-generated medical insights with real-time sources!")

# Search Bar
query = st.text_input("🔍 Enter a medical topic:")

if st.button("Search"):
    if query:
        st.subheader("📖 AI-Generated Medical Insights")
        response = generate_final_response(query)
        st.write(response)
        
        st.subheader("🎥 Relevant YouTube Videos")
        videos = search_youtube_videos(query)
        
        if videos:
            cols = st.columns(len(videos))
            for i, video in enumerate(videos):
                with cols[i]:
                    st.write(f"**{video['title']}**")
                    st.write(video["description"])
                    st.video(video["url"])
        else:
            st.warning("No videos found. Try another topic!")
    else:
        st.warning("Please enter a query.")
