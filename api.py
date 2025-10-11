from fastapi import FastAPI
from pydantic import BaseModel
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import textwrap
import numpy as np
import faiss

load_dotenv()

app = FastAPI()

client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("OPENAI_API_BASE")
)
deployment_name = os.getenv("OPENAI_DEPLOYMENT_NAME")

class SummaryRequest(BaseModel):
    text: str
    question: str = None

def chunk_text(text, chunk_size=500):
    return textwrap.wrap(text, chunk_size)

def get_embeddings(chunks):
    return [client.embeddings.create(
        model="text-embedding-ada-002",
        input=chunk
    ).data[0].embedding for chunk in chunks]

def build_faiss_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    return index

def retrieve_chunks(question, chunks, index):
    question_embedding = client.embeddings.create(
        model="text-embedding-ada-002",
        input=question
    ).data[0].embedding
    D, I = index.search(np.array([question_embedding]).astype('float32'), k=3)
    return [chunks[i] for i in I[0]]

@app.post("/summarize")
def summarize(request: SummaryRequest):
    chunks = chunk_text(request.text)
    embeddings = get_embeddings(chunks)
    index = build_faiss_index(embeddings)

    if request.question:
        relevant_chunks = retrieve_chunks(request.question, chunks, index)
        context = "\n\n".join(relevant_chunks)
        prompt = f"Using the following context, answer the question:\n\n{context}\n\nQuestion: {request.question}"
    else:
        context = request.text
        prompt = f"Summarize the following research paper:\n\n{context}"

    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=500
    )
    return {"response": response.choices[0].message.content}
