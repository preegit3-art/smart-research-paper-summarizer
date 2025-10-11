import streamlit as st
import pdfplumber
import requests
from io import BytesIO

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def query_fastapi(text, question=None):
    payload = {"text": text}
    if question:
        payload["question"] = question
    try:
        response = requests.post("https://smart-research-api.onrender.com/summarize", json=payload)
        data = response.json()
        return data.get("response", "No response received from the backend.")
    except Exception as e:
        return f" Error communicating with backend: {str(e)}"

st.set_page_config(page_title="Smart Summarizer", layout="wide")
st.title("Smart Research Paper Summarizer (FastAPI + RAG)")

uploaded_file = st.file_uploader("Upload a research paper (PDF)", type="pdf")
if uploaded_file:
    file_text = extract_text_from_pdf(BytesIO(uploaded_file.read()))
    st.text_area("Extracted Text Preview", file_text[:2000], height=300)

    if st.button("Summarize"):
        with st.spinner("Summarizing via FastAPI..."):
            summary = query_fastapi(file_text)
            st.subheader("Summary")
            st.write(summary)

    question = st.text_input("Ask a question about the paper:")
    if st.button("Get Answer"):
        with st.spinner("Answering via RAG..."):
            answer = query_fastapi(file_text, question)
            st.subheader("Answer")
            st.write(answer)

