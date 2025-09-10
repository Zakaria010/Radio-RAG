import os
import argparse
import pdfplumber
import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.retrieval import retrieve_relevant_chunks
from utils.generation import generate_answer
from pathlib import Path


DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_TOP_K = 5
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 100
DEFAULT_INDEX_TYPE = "innerproduct"

def extract_text_from_pdfs(pdf_folder):
    """
    Extract text from all PDF files in a folder.
    """
    texts = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            with pdfplumber.open(os.path.join(pdf_folder, pdf_file)) as pdf:
                text = "\n".join(
                    [page.extract_text() for page in pdf.pages if page.extract_text()]
                )
                texts.append(text)
    return texts

def chunk_text(texts, chunk_size, overlap, folder_path):
    """
    Split text into overlapping chunks.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    chunks_path = folder_path / f"chunks_{chunk_size}_{overlap}.npy"
    np.save(chunks_path, chunks)
    return chunks

def create_faiss_index(chunks, index_type, folder_path, embedder_name):
    """
    Create a FAISS index based on the selected type.
    """
    embedder = SentenceTransformer(embedder_name)
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]

    if index_type == "flatl2":
        index = faiss.IndexFlatL2(dimension)
    elif index_type == "innerproduct":
        index = faiss.IndexFlatIP(dimension)
    elif index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dimension, 32)  # HNSW with 32 connections per node
    elif index_type == "ivfflat":
        nlist = 100
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        index.train(embeddings)
    elif index_type == "ivfpq":
        nlist = 100
        m = 8  # Number of subquantizers
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)
        index.train(embeddings)
    elif index_type == "ivfsq":
        nlist = 100
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFScalarQuantizer(quantizer, dimension, nlist, faiss.ScalarQuantizer.QT_fp16)
        index.train(embeddings)
    else:
        raise ValueError(f"Unsupported index type: {index_type}")

    index.add(embeddings)
    index_path = folder_path / f"index_{index_type}.idx"
    faiss.write_index(index, str(index_path))
    print(f"✅ FAISS Index ({index_type}) and text chunks saved successfully.")


def load_model(model_name):
    """
    Load the generative model and its tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def prepare_index_and_chunks(pdf_folder, chunk_size, overlap, index_type, embedder_name):
    """
    Prepare (or create if necessary) the FAISS index and text chunks from PDFs.
    The folder is named based on the parameters, similar to evaluate_rag.
    """
    folder_name = f"{embedder_name} ; {index_type}_chunk{chunk_size}_overlap{overlap}"
    folder_path = Path(folder_name)
    if folder_path.exists():
        faiss_index_path = str(folder_path / f"index_{index_type}.idx")
        chunks_path = folder_path / f"chunks_{chunk_size}_{overlap}.npy"
    else:
        folder_path.mkdir(parents=True, exist_ok=True)
        texts = extract_text_from_pdfs(pdf_folder)
        chunks = chunk_text(texts, chunk_size, overlap, folder_path)
        create_faiss_index(chunks, index_type, folder_path, embedder_name)
        faiss_index_path = str(folder_path / f"index_{index_type}.idx")
        chunks_path = folder_path / f"chunks_{chunk_size}_{overlap}.npy"
    return faiss_index_path, chunks_path


def rag_agent(pdf_folder, chunk_size, overlap, index_type, model_name, k):
    """
    Interactive RAG chatbot that creates the FAISS index and text chunks if they don't exist.
    """
    print("\n📡 Telecom Regulation RAG Agent (type 'exit' to quit)\n")

    # Use the same embedder as in evaluate_rag for consistency
    embedder_name = "all-MiniLM-L6-v2"
    embedder = SentenceTransformer(embedder_name)
    faiss_index, chunks_path = prepare_index_and_chunks(pdf_folder, chunk_size, overlap, index_type, embedder_name)
    model, tokenizer = load_model(model_name)

    while True:
        query = input("Ask a question: ")
        if query.lower() == "exit":
            print("Exiting...")
            break

        retrieved_chunks = retrieve_relevant_chunks(query,embedder, k, faiss_index, chunks_path)
        answer = generate_answer(query, retrieved_chunks, model, tokenizer)

        print("\n🔹 Question:\n", query, "\n")
        print("\n💡 Answer:\n", answer, "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the interactive RAG agent with index creation from PDFs.")
    parser.add_argument("--pdf_folder", type=str, default="./data", help="Path to the folder containing PDF files.")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="Text chunk size.")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help="Overlap size between chunks.")
    parser.add_argument("--index_type", type=str, choices=["flatl2", "innerproduct", "hnsw", "ivfflat", "ivfpq", "ivfsq"],
                        default=DEFAULT_INDEX_TYPE, help="Type of FAISS index to use.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL, help="Hugging Face model name.")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Number of retrieved text chunks to use.")

    args = parser.parse_args()
    rag_agent(args.pdf_folder, args.chunk_size, args.overlap, args.index_type, args.model_name, args.top_k)
