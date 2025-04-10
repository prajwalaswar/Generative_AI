import getpass
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)

st.title('Langchain Demo With Gemini (language translator)')

input_text = st.text_input("Write the sentence in English and it will be translated into German")

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(
        chain.invoke(
            {
                "input_language": "English",
                "output_language": "German",
                "input": input_text,
            }
        )
    )