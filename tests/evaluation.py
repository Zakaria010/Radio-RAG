import os
import re
import argparse
import numpy as np
import faiss
import torch
import json
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from utils.retrieval import retrieve_relevant_chunks
from utils.generation import generate_answer, generate_answer_chat, generate_norag
from utils.reranker    import MetaLlamaReranker


DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
DEFAULT_TOP_K = 3
DEFAULT_CHUNK_SIZE = 400
DEFAULT_OVERLAP = 50
DEFAULT_INDEX_TYPE = "innerproduct"

# Load the JSON dataset
def load_my_dataset(path="regulations_qna_dataset.json"):
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    # Mark all entries as multiple-choice questions
    for entry in dataset:
        entry["type"] = "multiple-choice"
    return dataset

def parse_mc_answer(text):
    """
    Parse the generated text to extract the multiple‐choice answer letter.
    1) Look for 'Answer: A', 'Answer – A', 'Option A', or 'Option – A' (case‐insensitive).
    2) If not found, look for the first standalone letter A, B, C, or D.
    3) Return None if nothing matches.
    """
    # 2) Fallback: first standalone A/B/C/D
    match = re.search(r'\b([A-D])\b', text)
    if match:
        return match.group(1).upper()
   
    # 1) Try to catch "Answer: <LETTER>" or "Option <LETTER>"
    match = re.search(r'(?:Answer|Option)\s*[:\-]?\s*([A-D])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # 3) No valid letter found
    return None

def compute_mc_accuracy(ground_truth, generated_answer):
    """
    Given the true letter (ground_truth, e.g. "A") and the full generated text,
    extract the predicted letter and return 1.0 if it matches (case‐insensitive), else 0.0.
    """
    predicted = parse_mc_answer(generated_answer)
    if predicted is None:
        return 0.0
    if isinstance(ground_truth,list):
        return 1 if (predicted in ground_truth) else 0.0
    return 1.0 if predicted == ground_truth.upper() else 0.0

# Extract text from all PDFs in a folder
def extract_text_from_pdfs(pdf_folder):
    texts = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            try:
                import pdfplumber
            except ImportError:
                raise ImportError("Please install pdfplumber (pip install pdfplumber)")
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join(
                    [page.extract_text() for page in pdf.pages if page.extract_text()]
                )
                texts.append(text)
    return texts

# Split texts into overlapping chunks and save them
def chunk_text(texts, chunk_size, overlap, folder_path):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    chunks_path = folder_path / f"chunks_{chunk_size}_{overlap}.npy"
    np.save(chunks_path, chunks)
    return chunks

# Create and save a FAISS index of the chunks
def create_faiss_index(chunks, index_type, folder_path, embedder_name):
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
        index = faiss.IndexIVFScalarQuantizer(
            quantizer, dimension, nlist, faiss.ScalarQuantizer.QT_fp16
        )
        index.train(embeddings)
    else:
        raise ValueError(f"Unsupported index type: {index_type}")

    index.add(embeddings)
    index_path = folder_path / f"index_{index_type}.idx"
    faiss.write_index(index, str(index_path))
    print(f"✅ FAISS Index ({index_type}) and text chunks saved.")

# Prepare FAISS index and text chunks, reusing if already exists
def prepare_faiss_index(pdf_folder, chunk_size, overlap, index_type, embedder_name):
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

# Update or create a LaTeX table with accuracy results
def update_accuracy_latex_file(rag_parameters, accuracy_stats, file_path="ablmodelsgoodv1paperv2.txt"):
    new_rows = (
        f"\\textbf{{idx}} {rag_parameters['index_type']} & "
        f"\\multirow{{6}}{{*}}{{{accuracy_stats['accuracy']}}} \\\\ \\cline{{1-1}}\n"
    )
    for param, value in list(rag_parameters.items())[1:]:
        new_rows += f"\\textbf{{{param}}} {value} & \\\\ \\cline{{1-1}}\n"
    new_rows += "\\hline\n"

    if os.path.exists(file_path):
        with open(file_path, "r+", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "\\end{tabular}" in line:
                    lines.insert(i, new_rows)
                    break
            f.seek(0)
            f.writelines(lines)
            f.truncate()
    else:
        latex_table = (
            "\\begin{table*}[ht]\n\\centering\n"
            "\\begin{tabular}{|l|l|}\n\\hline\n"
            "RAG Parameters & Accuracy \\\\\n\\hline\n"
            + new_rows +
            "\\hline\n\\end{tabular}\n"
            "\\caption{Accuracy Scores and RAG Parameters}\n\\end{table*}"
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(latex_table)



def load_model(model_name, model_type):
    is_special = ("gemma" in model_name) or (model_name == "facebook/opt-13b")
    if is_special:
        cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "hub_models"))
        os.makedirs(cache_dir, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        if model_type.lower() == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir, device_map="auto")
        elif model_type.lower() == "causal":
            model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, device_map="auto")
        else:
            raise ValueError(f"Unknown model type '{model_type}'. Please choose 'seq2seq' or 'causal'.")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_type.lower() == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
        elif model_type.lower() == "causal":
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        else:
            raise ValueError(f"Unknown model type '{model_type}'. Please choose 'seq2seq' or 'causal'.")
    return model, tokenizer


# Main RAG evaluation function
def evaluate_rag(pdf_folder, chunk_size, overlap, index_type, model_name, k, norag,model_type,test_set, use_meta_llama_reranker=False, rerank_candidates=15):
    """
    Evaluate the RAG pipeline by generating answers for a set of multiple-choice questions.

    If norag is True, generation is done without context (baseline).
    Otherwise, context is retrieved via FAISS.
    """
    model, tokenizer = load_model(model_name,model_type)
    embedder_name = "all-MiniLM-L6-v2"
    embedder = SentenceTransformer(embedder_name)
    if use_meta_llama_reranker:
       print("Loading meta-llama reranker…")
       reranker = MetaLlamaReranker(
           model_name="meta-llama/Llama-3.2-3B-Instruct",
           device=f"cuda:{torch.cuda.current_device()}"
       )
    
    # Load the JSON dataset
    dataset = load_my_dataset(path=test_set)
    queries = [q["question"] for q in dataset]
    ground_truths = [q["answer"] for q in dataset]
    options_list = [q.get("options", []) for q in dataset]

    accuracies = []
    generated_answers = []
    all_run_accuracies = []
    num_runs = 1
    
    if norag:
        # Baseline without context retrieval
        for run in range(num_runs):
            run_accuracies = []
            print(f"\n[NoRAG] Run {run+1} Evaluation:\n")
            qnum = 1
            for query, ground_truth, options in zip(queries, ground_truths, options_list):
                print("processing question number : ", qnum,'\n')
                formatted_query = f"{query}\nOptions: " + " | ".join(options) + "\nPlease choose the correct option."
                generated_answer = generate_norag(formatted_query, model, tokenizer)
                acc = compute_mc_accuracy(ground_truth, generated_answer)
                run_accuracies.append(acc)
                print('correct answer : ', ground_truth, '\n')
                print('generated : ', generated_answer, '\n')
                print("accuracy : ", acc, '\n')
                generated_answers.append(generated_answer)
                qnum += 1

            run_avg = sum(run_accuracies) / len(run_accuracies)
            print(f"Run {run+1} Average Accuracy: {run_avg:.2f}")
            all_run_accuracies.append(run_avg)

        acc_array = np.array(all_run_accuracies)
        acc_mean = acc_array.mean()
        acc_std = acc_array.std()
        accuracy_stats = {"accuracy": f"{acc_mean:.2f} ± {acc_std:.2f}"}

        rag_parameters = {
            "index_type": "None",
            "chunks_size": 0,
            "overlap": 0,
            "top_k": 0,
            "model" : model_name,
            "norag": True
        }
        update_accuracy_latex_file(rag_parameters, accuracy_stats)
        return

    # RAG mode with FAISS retrieval
    faiss_index, chunks_path = prepare_faiss_index(pdf_folder, chunk_size, overlap, index_type, embedder_name)
    num_runs = 3  
    all_run_accuracies = []
    
    for run in range(num_runs):
        run_accuracies = []
        qnum = 1
        for query, ground_truth, options in zip(queries, ground_truths, options_list):
            print("processing question number : ", qnum,'\n')
            formatted_query = f"{query}\nOptions: " + " | ".join(options) + "\nPlease choose the correct option."
            initial = retrieve_relevant_chunks(formatted_query, embedder, k, faiss_index, chunks_path)
            if use_meta_llama_reranker:
                retrieved_chunks = reranker.rerank(
                    formatted_query,
                    initial,
                    top_n=k
                )
            else:
                retrieved_chunks = initial[:k]
            generated_answer = generate_answer(formatted_query, retrieved_chunks, model, tokenizer)
            acc = compute_mc_accuracy(ground_truth, generated_answer)
            print('correct answer : ', ground_truth, '\n')
            print('generated : ', generated_answer, '\n')
            print("accuracy : ", acc, '\n')
            run_accuracies.append(acc)
            generated_answers.append(generated_answer)
            qnum += 1
        
        run_avg = sum(run_accuracies) / len(run_accuracies)
        print(f"\nRun {run+1} Accuracy Evaluation:")
        for i, acc in enumerate(run_accuracies):
            print(f"Query {i+1}: Accuracy = {acc:.2f}")
        print(f"Run {run+1} Average Accuracy: {run_avg:.2f}")
        all_run_accuracies.append(run_avg)
    
    overall_avg = sum(all_run_accuracies) / len(all_run_accuracies)
    print(f"\nOverall Average Accuracy across {num_runs} runs: {overall_avg:.2f}")
    
    acc_array = np.array(all_run_accuracies)
    acc_mean = acc_array.mean()
    acc_std = acc_array.std()
    accuracy_stats = {"accuracy": f"{acc_mean:.2f} ± {acc_std:.2f}"}
    
    rag_parameters = {
        "index_type": index_type,
        "chunks_size": chunk_size,
        "overlap": overlap,
        "top_k": k,
        "model" : model_name,
        "norag" : False
    }
    update_accuracy_latex_file(rag_parameters, accuracy_stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate RAG pipeline with multiple-choice accuracy."
    )
    parser.add_argument("--pdf_folder", type=str, default="./data", help="Path to PDF folder.")
    parser.add_argument("--test", type=str, default="./data/QnA_set.json", help="Path to test dataset.")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help="Size of text chunks.")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP,
                        help="Overlap size between chunks.")
    parser.add_argument("--index_type", type=str,
                        choices=["flatl2","innerproduct","hnsw","ivfflat","ivfpq","ivfsq"],
                        default=DEFAULT_INDEX_TYPE, help="FAISS index type.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL,
                        help="Hugging Face model name.")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K,
                        help="Number of chunks to retrieve.")
    parser.add_argument("--norag", type=bool, default=False, help="Disable RAG.")
    parser.add_argument("--model_type", type=str, choices=["seq2seq","causal"],
                        default="causal", help="Model type: seq2seq or causal.")
    parser.add_argument(
        "--use_meta_llama_reranker",
        action="store_true",
        help="Apply Llama-3.2-3B reranking over FAISS hits"
    )
    parser.add_argument(
        "--rerank_candidates",
        type=int,
        default=15,
        help="How many FAISS hits to pull before reranking"
    )
    args = parser.parse_args()
    print(f"this script uses : {args.model_name}","\n")
    evaluate_rag(
        args.pdf_folder, args.chunk_size, args.overlap,
        args.index_type, args.model_name,
        args.top_k, args.norag, args.model_type, args.test,
        use_meta_llama_reranker=args.use_meta_llama_reranker,
        rerank_candidates=args.rerank_candidates
    )
