import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "facebook/opt-1.3b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
)

def generate_answer_chat(query, options, retrieved_chunks, model=model, tokenizer=tokenizer):
    """
    Generates an answer using the retrieved context, formatted as a conversation
    to better suit Llama 2 7B Chat's conversational tuning.
    """
    # Format each retrieved chunk as a numbered paragraph.
    paragraphs = [f"Paragraph {idx+1}: {chunk}" for idx, chunk in enumerate(retrieved_chunks)]
    context = "\n\n".join(paragraphs)
    
    # Create a conversational prompt.
    system_message = (
        "System: You are a telecom regulations expert. Answer using the information provided in the context. Start directly by Giving the best choice from options"
    )
    context_message = f"Context:\n{context}"
    user_message = f"User: {query}\nOptions: " + " | ".join(options)
    assistant_cue = "Assistant: "
    
    prompt = "\n\n".join([system_message, context_message, user_message, assistant_cue])
    
    # Determine the model type: seq2seq or causal.
    model_type = "seq2seq" if getattr(model.config, "is_encoder_decoder", False) else "causal"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        num_return_sequences=1,
        no_repeat_ngram_size=2
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if model_type == "causal":
        # Attempt to extract only the assistant's response.
        answer_start = generated_text.find("Assistant:")
        if answer_start != -1:
            answer = generated_text[answer_start + len("Assistant:"):].strip()
        else:
            answer = generated_text[len(prompt):].strip()
        return answer
    else:
        return generated_text.strip()


def generate_answer(query, retrieved_chunks, model=model, tokenizer=tokenizer):
    """
    Generates an answer using the retrieved context.

    For causal models, the prompt is included in the output so it must be removed.
    For seq2seq models, the output is directly the generated answer.
    """
    # Format each chunk as a separate paragraph with a numbered prefix.
    paragraphs = [f"Paragraph {idx+1}: {chunk}" for idx, chunk in enumerate(retrieved_chunks)]
    context = "\n\n".join(paragraphs)
    
    prompt = (f"You are a telecom regulations expert. Using the following context, answer the question:\n\n"
              f"Context:\n{context}\n\n"
              f"Question: {query}\nAnswer:")
              
    model_type = "seq2seq" if getattr(model.config, "is_encoder_decoder", False) else "causal"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs, 
        max_new_tokens=2048,  # Specifies the number of tokens to generate.
        num_return_sequences=1,
        no_repeat_ngram_size=2
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # For causal models, remove the prompt from the output.
    if model_type == "causal":
        # Remove the prompt from the output for causal models
        return generated_text[len(prompt):].strip()
    else:
        return generated_text.strip()


def generate_norag(query, model, tokenizer):
    """
    Generates an answer without additional context.
    """
    prompt = f"Answer the question:\n\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generate output with a specified maximum number of new tokens.
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,  # Specifies the number of tokens to generate.
        num_return_sequences=1,
        no_repeat_ngram_size=2
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True) 
    model_type = "seq2seq" if getattr(model.config, "is_encoder_decoder", False) else "causal"   
    
    if model_type == "causal":
        return generated_text[len(prompt):].strip()
    else:  # For seq2seq models
        return generated_text.strip()


