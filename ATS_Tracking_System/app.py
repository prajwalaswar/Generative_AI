from dotenv import load_dotenv
import streamlit as st
import os
import PyPDF2
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from the uploaded PDF
def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            pdf_text = ""
            for page in pdf_reader.pages:
                pdf_text += page.extract_text() or ""  # Handle pages without text gracefully
            if not pdf_text.strip():
                raise ValueError("No text could be extracted from the PDF.")
            return pdf_text
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return None
    else:
        st.error("No file uploaded")
        return None

def get_gemini_response(input_text, pdf_text, prompt):
    model = genai.GenerativeModel('gemini-pro')
    combined_input = f"Job Description: {input_text}\n\nResume Content: {pdf_text}\n\n{prompt}"
    response = model.generate_content([combined_input])
    return response.text

st.set_page_config(page_title="ATS Resume Expert", layout="centered")
st.header("ATS Tracking System")

input_text = st.text_area("Job Description:", key="input")
uploaded_file = st.file_uploader("Upload your resume (PDF)...", type=["pdf"])

submit1 = st.button("Tell Me About the Resume")
submit3 = st.button("Percentage Match")

input_prompt1 = """
You are an experienced Technical Human Resource Manager. Your task is to review the provided resume against the job description. 
Please share your professional evaluation on whether the candidate's profile aligns with the role. 
Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
"""

input_prompt3 = """
You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality. 
Your task is to evaluate the resume against the provided job description. 
Provide a percentage match, list the missing keywords, and offer final thoughts on alignment.
"""

if submit1:
    if uploaded_file:
        pdf_text = input_pdf_setup(uploaded_file)
        if pdf_text:
            response = get_gemini_response(input_text, pdf_text, input_prompt1)
            st.subheader("Response:")
            st.write(response)
        else:
            st.error("Could not process the uploaded PDF.")
    else:
        st.error("Please upload the resume.")

elif submit3:
    if uploaded_file:
        pdf_text = input_pdf_setup(uploaded_file)
        if pdf_text:
            response = get_gemini_response(input_text, pdf_text, input_prompt3)
            st.subheader("Response:")
            st.write(response)
        else:
            st.error("Could not process the uploaded PDF.")
    else:
        st.error("Please upload the resume.")
