# utils/reranker.py
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class MetaLlamaReranker:
    """
    Reranks a list of text chunks by asking meta-llama/Llama-3.2-3B-Instruct
    to score each one from 0–10.
    """
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        device: str = "cuda"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.device    = device
        self.system_prompt = (
            "You are an expert at evaluating document relevance for search queries.\n"
            "Rate each document on a scale from 0 to 10 (integer only) based on how well it answers the query.\n"
            "Reply with exactly one integer."
        )

    def rerank(self, query: str, chunks: list[str], top_n: int = 3) -> list[str]:
        scored = []
        for idx, chunk in enumerate(chunks):
            if idx % 5 == 0:
                print(f"  → scoring document {idx+1}/{len(chunks)}")
            prompt = (
                f"{self.system_prompt}\n\n"
                f"Query: {query}\n\n"
                f"Document:\n{chunk}\n\n"
                "Score:"
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            out    = self.model.generate(
                **inputs,
                max_new_tokens=4,
                num_return_sequences=1,
                do_sample=False
            )
            score_txt = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
            m = re.search(r'\b(10|[0-9])\b', score_txt)
            score = int(m.group(1)) if m else 0
            scored.append((chunk, score))

        # sort descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scored[:top_n]]
