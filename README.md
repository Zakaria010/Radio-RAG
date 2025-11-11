# Radio-RAG

Retrieval-Augmented Generation (RAG) for **radio regulations** (e.g., ITU rules and spectrum management).  
Index regulation PDFs with FAISS, retrieve the most relevant passages, and generate grounded answers with an LLM.

<p align="center">
  <a href="https://arxiv.org/abs/2509.09651" target="_blank">
    <img alt="arXiv Paper" src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg">
  </a>
</p>

<p align="center">
  <!-- Prefer PNG for inline rendering; keep a PDF link as fallback -->
  <img alt="RAG pipeline overview (Fig. 2)" src="assets/fig2_rag_pipeline.png" width="720">
</p>



---

## Table of Contents

- [Overview](#overview)
- [Radio-RAG Custom GPT](#radio-rag-custom-gpt)
- [Features](#features)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Examples](#examples)
  - [Common Arguments](#common-arguments)
- [Experiments](#experiments)
- [Hugging Face (ZeroGPU)](#hugging-face-zerogpu)
- [Figures](#figures)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## Overview

**Radio-RAG** implements a practical RAG pipeline tailored for telecom/spectrum regulations:

1. **Ingest** regulation PDFs and split them into chunks.  
2. **Embed** those chunks and build a **FAISS** index.  
3. **Retrieve** the most relevant passages for a user question.  
4. **Generate** grounded answers with an LLM using the retrieved context.

---

## Radio-RAG Custom GPT

Prefer a chat interface instead of running the code locally?

Use the **Radio-RAG Custom GPT**, a specialized assistant built on top of this RAG pipeline:

- ✅ Backed by the same **PDF → chunks → FAISS** retrieval used in this repo  
- 📚 Tailored for **ITU Radio Regulations & spectrum management** use-cases  
- 📎 Answers are **grounded in the indexed documents**, with references to the relevant provisions  
- 🧪 Great for quick checks, exploration, and validating how RAG behaves before full deployment  

👉 [Open the Radio-RAG Custom GPT](https://chatgpt.com/g/g-687bac267f508191a97daf466eccfa50-radio-regulations-gpt)

---

## Features

- 🔎 **PDF → chunks → FAISS**: simple, configurable ingestion pipeline  
- 🧠 **Model-agnostic**: choose your embedding and LLM backends  
- ⚙️ **Tunable retrieval**: chunk size, overlap, index type, top-K  
- 🧪 **Experiment ready**: compare vanilla LLM vs. RAG-augmented runs

---

## Quick Start

### 1) Install

~~~bash
git clone https://github.com/Zakaria010/Radio-RAG.git
cd Radio-RAG

# (optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install -r requirements.txt
~~~

### 2) Add your PDFs

Create the `data/` folder (if it doesn’t exist) and put your regulation PDFs inside:

~~~text
data/
├─ 2400594-RR-Vol 1-E-A5.pdf
├─ 2400594-RR-Vol 2-E-A5.pdf
├─ 2400594-RR-Vol 3-E-A5.pdf
├─ 2400594-RR-Vol 4-E-A5.pdf
└─ your_other_regulation_book.pdf
~~~

---

## Project Structure

~~~text
Radio-RAG/
├─ data/                 # Put regulation PDFs here
├─ tests/                # Evaluation / experiment scripts
├─ utils/                # Helpers (parsing, chunking, indexing, retrieval)
├─ local_rag.py          # CLI entry-point
├─ requirements.txt
├─ LICENSE
└─ README.md
~~~

---

## Usage

> Run the built-in help to see the **exact** flags supported by your current version.
~~~bash
python local_rag.py --help
~~~

### Example

**A) Ask a question **

~~~bash
python local_rag.py \
  --pdf_folder ./data \
~~~
Then the app will run and you can ask your question



### Common Arguments

- `--pdf_folder` *(str, default: `./data`)* — directory of PDFs  
- `--chunk_size` *(int)* — chunk length used for text splitting  
- `--overlap` *(int)* — overlap between adjacent chunks  
- `--index_type` *(str)* — FAISS index (`flatl2`, `hnsw`, `ivfflat`, `ivfpq`, …)  
- `--model_name` *(str)* — LLM ID/name  
- `--top_k` *(int)* — number of retrieved chunks  

---

## Experiments

Evaluation utilities live in `tests/`. A typical pattern:

~~~bash
python -m tests.evaluation  --chunk_size 400 --overlap 50 --index_type innerproduct --top_k 5 --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" --test "splits/eval.json"
~~~

This evaluation.py script include a switch like `--norag` to compare **vanilla LLM** vs **RAG**.  

---

## Hugging Face (ZeroGPU)

Prefer a hosted demo? Try the app on **Hugging Face Spaces** (ZeroGPU spins up on demand):

<p align="center">
  <a href="https://huggingface.co/spaces/zakinho00/RegRAGapp" target="_blank">
    <img alt="Open the HF Space" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Open%20App-blue.svg" width="220">
  </a>
</p>

---

## Figures

**Figure 5 — Vanilla vs. RAG (Qualitative)**

<p align="center">
  <img alt="GPT-4o vs RAG with context (Fig. 5)" src="assets/fig5_vanilla_vs_rag.png" width="720">
</p>

---

## Troubleshooting

- **No/irrelevant answers** → Confirm PDFs parse correctly; try larger `--top_k`; adjust `--chunk_size` / `--overlap`  
- **Index performance** → Start with `flatl2` (baseline) or `hnsw` (fast). IVF variants can help at larger scale  
- **Changed models/params** → Rebuild the index to avoid stale vectors

---

## Citation

If you use this repository, **please cite the paper**:

- **Paper:** [arXiv](https://arxiv.org/abs/2509.09651)

```bibtex
@misc{kassimi2025retrieval,
      title={Retrieval-Augmented Generation for Reliable Interpretation of Radio Regulations}, 
      author={Zakaria El Kassimi and Fares Fourati and Mohamed-Slim Alouini},
      year={2025},
      eprint={2509.09651},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2509.09651}, 
}







