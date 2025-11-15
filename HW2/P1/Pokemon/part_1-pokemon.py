import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import os
import argparse

# Config
MODEL_NAMES = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B"
]
JSON_PATH = "/home/aduarte/Trustworthy_AI/HW2/pokemon_mcqa.json"
BATCH_SIZE = 16
N_GPUS = 4

# Utility: Extract the answer letter ("A) ..." -> "A") from generation
def extract_choice(text):
    for c in ["A)", "B)", "C)", "D)"]:
        if c in text:
            return c
    # fallback: look for A/B/C/D only
    for c in ["A", "B", "C", "D"]:
        if f"{c})" in text:
            return f"{c})"
    return None

# Loads full list of questions (not grouped)
def load_questions(path):
    with open(path, "r") as f:
        return json.load(f)

def check_free_form_answer(response, correct_answer):
    """
    Check if the model's free-form response matches the correct answer.
    Uses lowercase matching for better accuracy.
    """
    response_lower = response.strip().lower()
    correct_lower = correct_answer.strip().lower()
    return correct_lower in response_lower

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate models on Pokemon QA dataset")
    parser.add_argument(
        "--experiment_type",
        type=str,
        default="mcqa",
        choices=["mcqa", "free-form"],
        help="Type of experiment: 'mcqa' for multiple choice or 'free-form' for free-form generation"
    )
    args = parser.parse_args()
    
    experiment_type = args.experiment_type
    print(f"Running experiment type: {experiment_type}")
    
    n_gpus = torch.cuda.device_count()
    print(f"GPUs available: {n_gpus}")
    assert n_gpus >= N_GPUS, f"Require at least {N_GPUS} GPUs."
    all_questions = load_questions(JSON_PATH)
    logs_dir = os.path.join(os.path.dirname(JSON_PATH), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    # Group questions by trait for reporting
    traits = ["type1", "hp", "defense", "speed"]
    questions_by_trait = {t: [] for t in traits}
    for q in all_questions:
        t = q.get("trait", "").lower()
        if t in questions_by_trait:
            questions_by_trait[t].append(q)
    # Results
    results = {m: {t: 0.0 for t in traits} for m in MODEL_NAMES}
    for mi, mname in enumerate(tqdm(MODEL_NAMES, desc="Models", leave=True)):
        print(f"\nEvaluating {mname}...")
        device = f"cuda:{mi % N_GPUS}" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(mname, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            mname,
            device_map={"": device},
            torch_dtype="auto"
        )
        for trait in tqdm(traits, desc=f"{mname} traits", leave=False):
            questions = questions_by_trait[trait]
            correct = 0
            total_batches = (len(questions) + BATCH_SIZE - 1) // BATCH_SIZE if questions else 0
            for i in tqdm(range(0, len(questions), BATCH_SIZE), total=total_batches, desc=f"{mname} {trait}", leave=False):
                batch = questions[i:i+BATCH_SIZE]
                prompts = []
                for q in batch:
                    if experiment_type == "mcqa":
                        # MCQA format: include options and ask for letter choice
                        content = q["question"].strip() + "\n" + "\n".join(q["options"]) + "\nAnswer:"
                        messages = [
                            {"role": "system", "content": "You are a helpful assistant that answers multiple choice questions. Your answer should be in the format of A), B), C), D). Do not include any other text in your answer."},
                            {"role": "user", "content": content}
                        ]
                    else:  # free-form
                        # Free-form format: just the question
                        content = q["question"].strip()
                        messages = [
                            {"role": "system", "content": "You are a helpful assistant that answers questions directly and concisely."},
                            {"role": "user", "content": content}
                        ]
                    chat_str = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
                    prompts.append(chat_str)
                inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
                # Adjust max_new_tokens based on experiment type
                max_tokens = 8 if experiment_type == "mcqa" else 50
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
                for idx, q in enumerate(batch):
                    prompt_len = len(inputs['input_ids'][idx])
                    output_ids = outputs[idx][prompt_len:]
                    response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                    
                    # Annotate this answer in output JSON
                    if 'model_answers' not in q:
                        q['model_answers'] = {}
                    q['model_answers'][mname] = response
                    
                    # Check correctness based on experiment type
                    if experiment_type == "mcqa":
                        pred = extract_choice(response)
                        if pred is not None and pred in q["answer"]:
                            correct += 1
                    else:  # free-form
                        if "answer_full_writing" in q:
                            if check_free_form_answer(response, q["answer_full_writing"]):
                                correct += 1
            acc = correct / len(questions) if questions else 0.0
            results[mname][trait] = acc
            print(f"{trait}: {acc:.4f}")
        del model
        torch.cuda.empty_cache()
    # Output metrics table (both to console and txt)
    # Use experiment type in output file names
    out_json = os.path.join(os.path.dirname(JSON_PATH), f"pokemon_{experiment_type}_with_answers.json")
    with open(out_json, "w") as f:
        json.dump(all_questions, f, indent=2)
    metrics_file = os.path.join(logs_dir, f"accuracy_metrics_{experiment_type}.txt")
    table_lines = []
    header = f"{'Trait':>10s}" + ''.join([f" | {m.split('-')[-1]:7s}" for m in MODEL_NAMES])
    table_lines.append(header)
    for trait in traits:
        row = f"{trait:>10s}"
        for m in MODEL_NAMES:
            row += f" | {results[m][trait]*100:7.2f}"
        table_lines.append(row)
    print("\n===== Accuracy Table =====")
    for line in table_lines:
        print(line)
    with open(metrics_file, "w") as out:
        out.write("===== Accuracy Table =====\n")
        for line in table_lines:
            out.write(line + "\n")
    print(f"\nMetrics and full answers saved to\n  {metrics_file}\n  {out_json}\n")

if __name__ == "__main__":
    main()

