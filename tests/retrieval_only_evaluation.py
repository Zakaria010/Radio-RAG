import os
import argparse
import numpy as np
import faiss
import torch
import json
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from utils.retrieval import retrieve_relevant_chunks
import evaluate


DEFAULT_TOP_K = 3
DEFAULT_CHUNK_SIZE = 400
DEFAULT_OVERLAP = 50
DEFAULT_INDEX_TYPE = "innerproduct"

# --- Load the JSON dataset ---
def load_my_dataset(path="regulations_qna_dataset.json"):
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    # Ensure the dataset contains the "context" field for retrieval evaluation.
    return dataset

# --- Set up ROUGE-L computation using evaluate ---
rouge = evaluate.load("rouge")

def compute_rouge_l(candidates: list, reference: str) -> float:
    """
    Compute the ROUGE-L F1 score between candidate and reference using the evaluate library.
    """
    candidate_text = "\n".join(candidates)
    result = rouge.compute(predictions=[candidate_text], references=[reference])
    # Extract the F1 measure for rougeL
    return result["rouge1"]

# --- Functions for data preparation and FAISS index creation ---
def extract_text_from_pdfs(pdf_folder):
    """Extract text from all PDF files in a folder."""
    texts = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            try:
                import pdfplumber
            except ImportError:
                raise ImportError("Please install pdfplumber (pip install pdfplumber)")
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                texts.append(text)
    return texts

def chunk_text(texts, chunk_size, overlap, folder_path):
    """Split the text into chunks with overlap and save the result."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    
    chunks_path = folder_path / f"chunks_{chunk_size}_{overlap}.npy"
    np.save(chunks_path, chunks)
    return chunks

def create_faiss_index(chunks, index_type, folder_path, embedder_name):
    """Create and save the FAISS index according to the specified type."""
    embedder = SentenceTransformer(embedder_name)
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]

    if index_type == "flatl2":
        index = faiss.IndexFlatL2(dimension)
    elif index_type == "innerproduct":
        index = faiss.IndexFlatIP(dimension)
    elif index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dimension, 32)
    elif index_type == "ivfflat":
        nlist = 100
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        index.train(embeddings)
    elif index_type == "ivfpq":
        nlist = 100
        m = 8
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
    print(f"✅ FAISS Index ({index_type}) and text chunks saved.")

def prepare_faiss_index(pdf_folder, chunk_size, overlap, index_type, embedder_name):
    """
    Prepare the FAISS index and text chunks.
    If the folder already exists, use the existing files;
    otherwise, extract, split and create the index.
    """
    folder_name = f"{embedder_name} ; {index_type}_chunk{chunk_size}_overlap{overlap}"
    folder_path = Path(folder_name)
    
    if folder_path.exists():
        faiss_index = str(folder_path / f"index_{index_type}.idx")
        chunks_path = folder_path / f"chunks_{chunk_size}_{overlap}.npy"
    else:
        folder_path.mkdir(parents=True, exist_ok=True)
        texts = extract_text_from_pdfs(pdf_folder)
        chunks = chunk_text(texts, chunk_size, overlap, folder_path)
        create_faiss_index(chunks, index_type, folder_path, embedder_name)
        faiss_index = str(folder_path / f"index_{index_type}.idx")
        chunks_path = folder_path / f"chunks_{chunk_size}_{overlap}.npy"
    
    return faiss_index, chunks_path

def update_accuracy_latex_file(rag_parameters, accuracy_stats, file_path="retrieval_a.txt"):
    """Update (or create) a LaTeX table with the accuracy scores."""
    new_latex_rows = (
        f"\\textbf{{idx}} {rag_parameters['index_type']} & \\multirow{{5}}{{*}}{{{accuracy_stats['accuracy']}}} \\\\ \\cline{{1-1}}\n"
    )
    for param, value in list(rag_parameters.items())[1:]:
        new_latex_rows += f"\\textbf{{{param}}}  {value} & \\\\ \\cline{{1-1}}\n"
    new_latex_rows += "\\hline\n"
    
    if os.path.exists(file_path):
        with open(file_path, "r+", encoding="utf-8") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                if "\\end{tabular}" in lines[i]:
                    lines.insert(i, new_latex_rows)
                    break
            f.seek(0)
            f.writelines(lines)
            f.truncate()
    else:
        latex_table = (
            "\\begin{table*}[ht]\n\\centering\n\\begin{tabular}{|l|l|}\n\\hline\n"
            "RAG Parameters & Accuracy \\\\ \n\\hline\n"
            + new_latex_rows +
            "\\hline\n\\end{tabular}\n\\caption{Accuracy Scores and RAG Parameters}\n\\end{table*}"
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(latex_table)

# --- Evaluate the retrieval process ---
def evaluate_retrieval(pdf_folder, chunk_size, overlap, index_type, top_k, retrieval_threshold=0.7):
    """
    Evaluate the retrieval of chunks only.
    For each question in the dataset, retrieve chunks via FAISS and compare each retrieved chunk
    with the reference "context" using the ROUGE-L F1 score (via the evaluate library).
    If the maximum score >= retrieval_threshold, the accuracy for the question is 1.
    """
    embedder_name = "all-MiniLM-L6-v2"
    embedder = SentenceTransformer(embedder_name)
    
    # Prepare the FAISS index and load the text chunks
    faiss_index, chunks_path = prepare_faiss_index(pdf_folder, chunk_size, overlap, index_type, embedder_name)
    
    # Load the dataset (each entry must contain "question" and "context")
    dataset = load_my_dataset(path="test_dataset_gglxxl.json")
    queries = [q["question"] for q in dataset]
    contexts = [q["context"] for q in dataset]
    options_list = [q.get("options", []) for q in dataset]
    
    accuracies = []
    retrieval_scores = []
    qnum = 1
    
    for query, ref_context, options in zip(queries, contexts, options_list):
        print(f"Processing question {qnum}...")
        formatted_query = f"{query}\nOptions: " + " | ".join(options) + "\nPlease choose the correct option."
        retrieved_chunks = retrieve_relevant_chunks(formatted_query, embedder, top_k, faiss_index, chunks_path)
        
        # Compute ROUGE-L F1 score for each retrieved chunk against the reference context
        score = compute_rouge_l(retrieved_chunks, ref_context) 
        retrieval_scores.append(score)
        acc = 1 if score >= retrieval_threshold else 0
        accuracies.append(acc)
        
        print(f"Question: {query}")
        print(f"Max ROUGE-L F1 score: {score:.2f} (Threshold: {retrieval_threshold})")
        print(f"Accuracy for this question: {acc}\n")
        qnum += 1

    overall_accuracy = sum(accuracies) / len(accuracies)
    avg_retrieval_score = sum(retrieval_scores) / len(retrieval_scores)
    print(f"Overall Retrieval Accuracy: {overall_accuracy:.2f}")
    print(f"Average ROUGE-L F1 score across questions: {avg_retrieval_score:.2f}")

    # Optional: update the LaTeX file with the evaluation results
    rag_parameters = {
        "index_type": index_type,
        "chunks_size": chunk_size,
        "overlap": overlap,
        "top_k": top_k,
        "embedder": embedder_name,
        "retrieval_threshold": retrieval_threshold
    }
    accuracy_stats = {"accuracy": f"{overall_accuracy:.2f}"}
    update_accuracy_latex_file(rag_parameters, accuracy_stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the retrieval part by comparing the retrieved chunks with the reference context using ROUGE-L F1."
    )
    parser.add_argument("--pdf_folder", type=str, default="./data", help="Path to the folder containing PDF files.")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="Size of text chunks.")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help="Overlap size between chunks.")
    parser.add_argument("--index_type", type=str, choices=["flatl2", "innerproduct", "hnsw", "ivfflat", "ivfpq", "ivfsq"],
                        default=DEFAULT_INDEX_TYPE, help="Type of FAISS index to use.")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Number of retrieved chunks.")
    parser.add_argument("--threshold", type=float, default=0.7, help="ROUGE-L F1 threshold for a correct retrieval.")
    args = parser.parse_args()
    max_f1 = (2 * min(args.chunk_size * args.top_k, 400) ) / (400 + args.chunk_size * args.top_k)
    
    evaluate_retrieval(args.pdf_folder, args.chunk_size, args.overlap, args.index_type, args.top_k, max_f1 * args.threshold)
