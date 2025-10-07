import streamlit as st
import pdfplumber
import os
from dotenv import load_dotenv
from io import BytesIO
from openai import AzureOpenAI

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
        header_match = re.match(r"^(Abstract|Introduction|Methods?|Results?|Discussion|Conclusion|References?)$", line.strip(), re.IGNORECASE)
        if header_match:
            current_section = header_match.group(1).capitalize()
            sections[current_section] = ""
        else:
            sections[current_section] += line + "\n"
    return sections

# Summarize section using Azure OpenAI
def summarize_section(section_name, section_text):
    prompt = f"Summarize the following {section_name} section of a research paper in 3-4 sentences:\n\n{section_text}"
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=300
    )
    return response.choices[0].message.content

# Answer question using context
def answer_question(context, question):
    prompt = f"Using the following context from a research paper, answer the question. Cite the section if possible.\n\nContext:\n{context}\n\nQuestion: {question}"
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=300
    )
    return response.choices[0].message.content

# Streamlit UI
st.set_page_config(page_title="Smart Research Paper Summarizer", layout="wide")
st.title("Smart Research Paper Summarizer")

# Session state for storing papers
if "papers" not in st.session_state:
    st.session_state.papers = {}

# Upload PDFs
st.header("Upload Research Papers")
uploaded_files = st.file_uploader("Upload one or two PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        file_text = extract_text_from_pdf(BytesIO(file.read()))
        sections = split_into_sections(file_text)
        st.session_state.papers[file.name] = {
            "text": file_text,
            "sections": sections,
            "summaries": {sec: summarize_section(sec, txt) for sec, txt in sections.items()}
        }
    st.success("PDFs processed and summarized successfully!")

# Display summaries
if st.session_state.papers:
    st.header("Section-wise Summaries")
    selected_paper = st.selectbox("Select a paper to view summaries", list(st.session_state.papers.keys()))
    paper_data = st.session_state.papers[selected_paper]
    for section, summary in paper_data["summaries"].items():
        with st.expander(f" {section}"):
            st.write(summary)

# Q&A
if st.session_state.papers:
    st.header("Ask a Question")
    selected_paper_qna = st.selectbox("Select a paper for Q&A", list(st.session_state.papers.keys()), key="qna_paper")
    question = st.text_input("Enter your question about the paper:")
    if st.button("Get Answer"):
        context_text = st.session_state.papers[selected_paper_qna]["text"]
        answer = answer_question(context_text, question)
        st.markdown(f"**Answer:** {answer}")

# Compare papers
if len(st.session_state.papers) == 2:
    st.header("Compare Two Papers")
    paper_names = list(st.session_state.papers.keys())
    paper1, paper2 = paper_names[0], paper_names[1]
    st.subheader(f"Comparison: {paper1} vs {paper2}")
    for section in ["Abstract", "Introduction", "Methods", "Results", "Conclusion"]:
        sum1 = st.session_state.papers[paper1]["summaries"].get(section, "Not found")
        sum2 = st.session_state.papers[paper2]["summaries"].get(section, "Not found")
        with st.expander(f" {section} Comparison"):
            col1, col2 = st.columns(2)
            col1.markdown(f"**{paper1}**\n\n{sum1}")
            col2.markdown(f"**{paper2}**\n\n{sum2}")

