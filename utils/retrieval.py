import faiss
import numpy as np
from sentence_transformers import SentenceTransformer



def retrieve_relevant_chunks(query,embedder, k=5, index_path="faiss_index.idx", chunks_path="text_chunks.npy"):
    index = faiss.read_index(index_path)
    chunks = np.load(chunks_path, allow_pickle=True)
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)

    retrieved_chunks = [chunks[i] for i in indices[0]]
    return retrieved_chunks
