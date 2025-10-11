import streamlit as st
import pdfplumber
import os
from dotenv import load_dotenv
from io import BytesIO
from openai import AzureOpenAI
import faiss
import numpy as np
import textwrap

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("OPENAI_API_BASE")
)
deployment_name = os.getenv("OPENAI_DEPLOYMENT_NAME")

# Extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Split text into sections
def split_into_sections(text):
    import re
    sections = {}
    current_section = "Introduction"
    sections[current_section] = ""
    for line in text.splitlines():
        header_match = re.match(r"^(Abstract|Introduction)$", line.strip(), re.IGNORECASE)
        if header_match:
            current_section = header_match.group(1).capitalize()
            sections[current_section] = ""
        else:
            sections[current_section] += line + "\n"
    return sections

# Chunk text for RAG
def chunk_text(text, chunk_size=500):
    return textwrap.wrap(text, chunk_size)

# Generate embeddings for chunks
def get_embeddings(chunks):
    return [client.embeddings.create(
        model="text-embedding-ada-002",
        input=chunk
    ).data[0].embedding for chunk in chunks]

# Build FAISS index
def build_faiss_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    return index

# Retrieve relevant chunks for question
def retrieve_relevant_chunks(question, chunks, index):
    question_embedding = client.embeddings.create(
        model="text-embedding-ada-002",
        input=question
    ).data[0].embedding
    D, I = index.search(np.array([question_embedding]).astype('float32'), k=3)
    return [chunks[i] for i in I[0]]

# Generate answer using retrieved chunks
def answer_with_rag(chunks, question):
    context = "\n\n".join(chunks)
    prompt = f"Using the following context, answer the question:\n\n{context}\n\nQuestion: {question}"
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=300
    )
    return response.choices[0].message.content

# Streamlit UI
st.set_page_config(page_title="Smart Research Paper Summarizer with RAG", layout="wide")
st.title("Smart Research Paper Summarizer with RAG")

if "papers" not in st.session_state:
    st.session_state.papers = {}

st.header("Upload Research Papers")
uploaded_files = st.file_uploader("Upload one or two PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        file_text = extract_text_from_pdf(BytesIO(file.read()))
        sections = split_into_sections(file_text)

        total_summary_prompt = f"Summarize the entire research paper in 5-6 sentences:\n\n{file_text}"
        total_summary_response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": total_summary_prompt}],
            temperature=0.5,
            max_tokens=500
        )
        total_summary = total_summary_response.choices[0].message.content

        chunks = chunk_text(file_text)
        embeddings = get_embeddings(chunks)
        index = build_faiss_index(embeddings)

        st.session_state.papers[file.name] = {
            "text": file_text,
            "sections": sections,
            "total_summary": total_summary,
            "summaries": {sec: summarize_section(sec, txt) for sec, txt in sections.items()},
            "chunks": chunks,
            "index": index
        }
    st.success("PDFs processed and summarized successfully!")

if st.session_state.papers:
    st.header("Total Paper Summary")
    selected_paper = st.selectbox("Select a paper to view summary", list(st.session_state.papers.keys()))
    paper_data = st.session_state.papers[selected_paper]
    st.markdown(f"**Summary of {selected_paper}:**")
    st.write(paper_data["total_summary"])

if st.session_state.papers:
    st.header("Ask a Question (RAG-powered)")
    selected_paper_qna = st.selectbox("Select a paper for Q&A", list(st.session_state.papers.keys()), key="qna_paper")
    question = st.text_input("Enter your question about the paper:")
    if st.button("Get Answer"):
        chunks = st.session_state.papers[selected_paper_qna]["chunks"]
        index = st.session_state.papers[selected_paper_qna]["index"]
        relevant_chunks = retrieve_relevant_chunks(question, chunks, index)
        answer = answer_with_rag(relevant_chunks, question)
        st.markdown(f"**Answer:** {answer}")

if len(st.session_state.papers) == 2:
    st.header("Compare Two Papers")
    paper_names = list(st.session_state.papers.keys())
    paper1, paper2 = paper_names[0], paper_names[1]
    st.subheader(f"Comparison: {paper1} vs {paper2}")
    for section in ["Abstract", "Introduction"]:
        sum1 = st.session_state.papers[paper1]["summaries"].get(section, "Not found")
        sum2 = st.session_state.papers[paper2]["summaries"].get(section, "Not found")
        with st.expander(f" {section} Comparison"):
            col1, col2 = st.columns(2)
            col1.markdown(f"**{paper1}**\n\n{sum1}")

