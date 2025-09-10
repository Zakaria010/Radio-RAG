import os
import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdfs(pdf_folder):
    texts = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            with pdfplumber.open(os.path.join(pdf_folder, pdf_file)) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                texts.append(text)
    return texts

def chunk_text(texts, chunk_size=500, overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return chunks

def create_faiss_index(chunks, index_path="faiss_index.idx", chunks_path="text_chunks.npy"):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(chunks, convert_to_numpy=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)  

    faiss.write_index(index, index_path)
    np.save(chunks_path, chunks)

    print("FAISS Index and text chunks saved successfully.")

if __name__ == "__main__":
    pdf_folder = "./data"
    texts = extract_text_from_pdfs(pdf_folder)
    chunks = chunk_text(texts)
    create_faiss_index(chunks)
