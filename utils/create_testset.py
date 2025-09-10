import os
import json
import torch
import gc
import re
import pdfplumber
from transformers import pipeline


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using pdfplumber."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def process_pdf_directory(pdf_dir):
    """Extract text from all PDF files in the given directory."""
    all_text = ""
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            print(f"Processing {pdf_path}...")
            file_text = extract_text_from_pdf(pdf_path)
            all_text += file_text + "\n"
    return all_text


def split_text_into_chunks(text, chunk_size=300):
    """Split the text into chunks of a given number of words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def filter_noise_lines(text, max_digit_ratio=0.5):
    """Drop lines where digits+punctuation exceed max_digit_ratio of total chars."""
    clean_lines = []
    for line in text.splitlines():
        # If the line is mostly digits/punct, skip it
        non_ws = line.replace(" ", "")
        if not non_ws:
            continue
        digit_count = len(re.findall(r"[0-9]", non_ws))
        punct_count = len(re.findall(r"[^\w\s]", non_ws))
        if (digit_count + punct_count) / len(non_ws) > max_digit_ratio:
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines)

def strip_headers_footers(text):
    # Adjust the regex to match your specific boilerplate
    text = re.sub(r"^Page\s+\d+\s+of\s+\d+(\r?\n)?", "", text, flags=re.MULTILINE)
    # Remove any lines like “Company Confidential” or “© 2024 Telecom Corp”
    text = re.sub(r".*Confidential.*|©\s*\d{4}.*", "", text)
    return text

def normalize_whitespace(text):
    text = re.sub(r"\r\n?", "\n", text)           # unify line breaks
    text = re.sub(r"\n{2,}", "\n\n", text)         # at most one blank line
    text = re.sub(r"[“”]", '"', text)              # normalize quotes
    return text.strip()

def clean_and_chunk(text, chunk_size=300):
    text = strip_headers_footers(text)
    text = filter_noise_lines(text)
    text = normalize_whitespace(text)
    # optionally: text = keep_relevant_sentences(text)
    
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


generator = pipeline("text2text-generation", model="google/flan-t5-xxl")

def generate_qna_from_chunk(chunk, prompt_template):
    """Generate QnA text from a text chunk using a prompt template."""
    prompt = prompt_template.format(context=chunk)
    outputs = generator(prompt, max_length=500, num_return_sequences=1, no_repeat_ngram_size=2)
    qna_text = outputs[0]['generated_text']
    # Debug print
    # print("Generated QnA text:\n", qna_text)
    return qna_text

def split_options(options_string):
    """
    Splits a single string of options (marked by A, B, C, D)
    into a list of option texts, preserving the markers.
    Example:
      "A Option1 text.B Option2 text.C Option3 text.D Option4 text."
      returns: ['A Option1 text.', 'B Option2 text.', 'C Option3 text.', 'D Option4 text.']
    """
    pattern = r'([ABCD])\s*(.*?)(?=(?:[ABCD]\s)|$)'
    matches = re.findall(pattern, options_string)
    return [f"{marker} {text.strip()}" for marker, text in matches]


def parse_qna_text(qna_text, chunk):
    """
    Parse the QnA text to extract questions, options, answer, and explanation,
    and return a list of dataset entries.
    """
    dataset = []
    # Regex to capture question, options, answer, and optional explanation.
    pattern = r"Q:\s*(?P<question>.*?)\s*Options:\s*(?P<options>.*?)\s*Answer:\s*(?P<answer>.*?)(?:\s*Explanation:\s*(?P<explanation>.*?))?(?=\s*Q:|$)"
    matches = re.finditer(pattern, qna_text, re.DOTALL)
    for match in matches:
        question = match.group("question").strip()
        options_str = match.group("options").strip()
        options_pattern = r"A\)\s*(?P<A>.*?)(?=\s*B\))\s*B\)\s*(?P<B>.*?)(?=\s*C\))\s*C\)\s*(?P<C>.*?)(?=\s*D\))\s*D\)\s*(?P<D>.*)"
        options_match = re.search(options_pattern, options_str, re.DOTALL)
        if options_match:
            options = [options_match.group(letter).strip() for letter in ['A', 'B', 'C', 'D']]
        else:
            options = [opt.strip() for opt in options_str.split("|") if opt.strip()]
        answer = match.group("answer").strip()
        explanation = match.group("explanation").strip() if match.group("explanation") else ""
        dataset.append({
            "question": question,
            "options": options,
            "answer": answer,
            "explanation": explanation,
            "context": chunk
        })
    new_dataset = []
    for item in dataset:
        new_item = item.copy()
        if 'options' in new_item and new_item['options']:
            options_str = new_item['options'][0]
            new_item['options'] = split_options(options_str)
        new_dataset.append(new_item)
    return new_dataset


def judge_qna_entry(entry, judge_evaluator, judge_prompt_template):
    """
    This function uses a specialized telecom LLM (tele-llm-de-quali) to evaluate a generated QnA entry.
    It returns True if the judge deems the entry as 'good' (i.e. high quality and consistent with the context),
    otherwise False.
    """
    # Build the judge prompt using the provided template and entry details
    judge_prompt = judge_prompt_template.format(
        context=entry["context"],
        question=entry["question"],
        options=' | '.join(entry["options"]),
        answer=entry["answer"],
        explanation=entry["explanation"]
    )
    # Get the judge's output
    judge_output = judge_evaluator(judge_prompt, max_length=2048, do_sample=False)
    judgment = judge_output[0]['generated_text'].strip()
    print("Judge output:", judgment)
    # Accept the entry if the judgment contains "good" (case-insensitive)
    return "good" in judgment.lower()


def generate_qna_dataset(chunks, prompt_template, judge_evaluator, judge_prompt_template, desired_num_questions=150):
    """
    Generate a QnA dataset from text chunks using a generation model and a judge evaluator.
    """
    final_dataset = []
    total_chunks = len(chunks)
    if total_chunks == 0:
        return final_dataset

    # Calculate the step size to evenly sample chunks
    step = max(1, total_chunks // desired_num_questions)
    processed_indices = set()

    # First pass: process chunks at indices 0, step, 2*step, ...
    for i in range(0, total_chunks, step):
        try:
            print(f"Processing chunk {i} (first pass)...")
            qna_text = generate_qna_from_chunk(chunks[i], prompt_template)
            qna_entries = parse_qna_text(qna_text, chunks[i])
            # Evaluate each QnA entry using the judge evaluator
            for entry in qna_entries:
                if judge_qna_entry(entry, judge_evaluator, judge_prompt_template):
                    final_dataset.append(entry)
                    if len(final_dataset) >= desired_num_questions:
                        return final_dataset[:desired_num_questions]
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")
        processed_indices.add(i)

    # Second pass: try additional offsets if not enough questions have been generated
    for offset in range(1, step):
        for i in range(offset, total_chunks, step):
            if i in processed_indices:
                continue
            try:
                print(f"Processing chunk {i} (offset {offset} pass)...")
                qna_text = generate_qna_from_chunk(chunks[i], prompt_template)
                qna_entries = parse_qna_text(qna_text, chunks[i])
                for entry in qna_entries:
                    if judge_qna_entry(entry, judge_evaluator, judge_prompt_template):
                        final_dataset.append(entry)
                        if len(final_dataset) >= desired_num_questions:
                            return final_dataset[:desired_num_questions]
            except Exception as e:
                print(f"Error processing chunk {i}: {e}")
    return final_dataset


def evaluate_llm_on_dataset(dataset, evaluator):
    """
    Evaluate the generated QnA dataset using a given evaluator LLM.
    Returns the accuracy as a percentage.
    """
    correct_count = 0
    for entry in dataset:
        prompt = (
            f"Question: {entry['question']}\n"
            f"Options: {' | '.join(entry['options'])}\n"
            "Please select the correct option from the ones above."
        )
        print(prompt, '\n')
        output = evaluator(prompt, max_length=100, do_sample=False)
        llm_answer = output[0]['generated_text'].strip()
        print(llm_answer, '\n')
        if entry['answer'].lower() in llm_answer.lower():
            correct_count += 1
            print(correct_count)
    accuracy = (correct_count / len(dataset)) * 100 if dataset else 0
    return accuracy

# ----- Main execution -----
if __name__ == "__main__":
    # Specify the directory containing your PDF files
    pdf_directory = "./data"  
    print("Extracting text from PDFs...")
    all_text = process_pdf_directory(pdf_directory)
    print("Text extraction complete.")

    chunks = clean_and_chunk(all_text, chunk_size=300)
    print(f"Text split into {len(chunks)} chunks.")

    # Define a prompt template for question generation.
    prompt_template = (
        "You are a telecom regulations expert. Given the following telecom regulations excerpt, generate "
        "ONE multiple-choice question with 4 options. Ensure that your output follows this exact format:\n\n"
        "Example:\n"
        "Q: What is the role of the ITU in spectrum management?\n"
        "Options: A) Allocating spectrum | B) Enforcing national laws | C) Manufacturing telecom equipment | D) Regulating ISPs\n"
        "Answer: A) Allocating spectrum\n"
        "Explanation: The ITU is responsible for managing global spectrum allocation and ensuring fair use.\n\n"
        "Context: {context}\n\n"
        "You should generate a question, 4 options, the correct answer, and an explanation for your choice. Ensure that your output follows this exact format.\n\n"
    )

    # Define a prompt template for the judge evaluation.
    judge_prompt_template = (
        "You are an impartial judge specialized in telecom regulations quality evaluation. Given the following context and generated QnA, "
        "determine if the question is of high quality and fully consistent with the context. "
        "Respond only with 'Good' if the QnA entry meets these criteria, otherwise respond with 'Bad'.\n\n"
        "Context: {context}\n\n"
        "Question: {question}\n"
        "Options: {options}\n"
        "Answer: {answer}\n"
        "Explanation: {explanation}\n\n"
        "Your evaluation:"
    )

    
    judge_evaluator = pipeline("text-generation", model="AliMaatouk/LLama-3-8B-Tele-it")
    
    dataset = generate_qna_dataset(chunks, prompt_template, judge_evaluator, judge_prompt_template, desired_num_questions=1500)
    print(f"Generated {len(dataset)} questions after judge evaluation.")

    # Save the generated dataset to a JSON file
    with open("try_qna_dataset_cleaned_v1500.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print("Dataset saved to regulations_qna_big_dataset.json.")

    del generator, judge_evaluator  
    gc.collect()
    torch.cuda.empty_cache()

    evaluator = pipeline("text2text-generation", model="google/flan-t5-xxl")
    accuracy = evaluate_llm_on_dataset(dataset, evaluator)
    print(f"LLM Accuracy on the generated dataset: {accuracy:.2f}%")