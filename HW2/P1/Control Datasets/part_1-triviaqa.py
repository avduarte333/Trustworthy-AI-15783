import os
import sys
import json
import re
import string
from typing import List, Dict, Tuple
from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Configuration
MODEL_NAMES = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
]
N_GPUS = 4
BATCH_SIZE = 8
MAX_NEW_TOKENS = 256
MAX_RETRIES = 3

OUTPUT_ROOT = "/home/aduarte/Trustworthy_AI/HW2/logs/triviaqa"


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing and removing punctuation."""
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


def tokenize(text: str) -> List[str]:
    """Tokenize text into list of tokens."""
    return text.split()


def exact_match_score(prediction: str, gold_aliases: List[str]) -> int:
    """
    Compute exact match: 1 if any normalized alias is contained in prediction, else 0.
    """
    pred_norm = normalize_text(prediction)
    for alias in gold_aliases:
        alias_norm = normalize_text(alias)
        if alias_norm in pred_norm or pred_norm in alias_norm:
            return 1
    return 0


def f1_score(prediction_tokens: List[str], gold_tokens: List[str]) -> float:
    """Compute token-level F1 score between prediction and gold."""
    if len(gold_tokens) == 0 or len(prediction_tokens) == 0:
        return float(len(gold_tokens) == 0 and len(prediction_tokens) == 0)
    
    common = Counter(prediction_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


def recall_score(prediction_tokens: List[str], gold_tokens: List[str]) -> float:
    """Compute token-level recall between prediction and gold."""
    if len(gold_tokens) == 0:
        return 1.0 if len(prediction_tokens) == 0 else 0.0
    
    common = Counter(prediction_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    
    recall = num_same / len(gold_tokens)
    return recall


def max_f1_recall(prediction: str, gold_aliases: List[str]) -> Tuple[float, float]:
    """
    Compute max F1 and Recall over all gold aliases.
    """
    pred_tokens = tokenize(normalize_text(prediction))
    
    max_f1 = 0.0
    max_recall = 0.0
    
    for alias in gold_aliases:
        gold_tokens = tokenize(normalize_text(alias))
        f1 = f1_score(pred_tokens, gold_tokens)
        recall = recall_score(pred_tokens, gold_tokens)
        max_f1 = max(max_f1, f1)
        max_recall = max(max_recall, recall)
    
    return max_f1, max_recall


def compute_metrics(prediction: str, gold_aliases: List[str]) -> Dict[str, float]:
    """Compute EM, F1, and Recall metrics."""
    em = exact_match_score(prediction, gold_aliases)
    f1, recall = max_f1_recall(prediction, gold_aliases)
    return {"em": em, "f1": f1, "recall": recall}


def build_chat_prompt(tokenizer, question: str) -> str:
    """Build chat prompt for the model."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the following question concisely "
                "with a brief factual answer."
            ),
        },
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate_responses_for_model(model_name: str, device: str, inputs: List[Dict]) -> Dict[str, str]:
    """Generate responses for a single model with retry logic."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": device},
        dtype="auto",
    )
    model.eval()

    question_texts = [ex["question"] for ex in inputs]
    question_to_response: Dict[str, str] = {}

    total_batches = (len(question_texts) + BATCH_SIZE - 1) // BATCH_SIZE if question_texts else 0
    for i in tqdm(range(0, len(question_texts), BATCH_SIZE), total=total_batches, desc=f"{model_name} batches", leave=False):
        batch_questions = question_texts[i : i + BATCH_SIZE]
        
        # Try generation with retries
        retries = 0
        while retries < MAX_RETRIES:
            try:
                chat_batch = [build_chat_prompt(tokenizer, q) for q in batch_questions]
                inputs_enc = tokenizer(
                    chat_batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs_enc,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                    )
                
                for idx, q in enumerate(batch_questions):
                    prompt_len = len(inputs_enc["input_ids"][idx])
                    out_ids = outputs[idx][prompt_len:]
                    text = tokenizer.decode(out_ids, skip_special_tokens=True).strip()
                    question_to_response[q] = text
                
                break  # Success, exit retry loop
                
            except Exception as e:
                retries += 1
                if retries >= MAX_RETRIES:
                    print(f"Failed after {MAX_RETRIES} retries for batch at index {i}: {e}")
                    # Assign empty string for failed batches
                    for q in batch_questions:
                        if q not in question_to_response:
                            question_to_response[q] = ""
                else:
                    print(f"Error on attempt {retries}/{MAX_RETRIES}: {e}, retrying...")

    # Cleanup
    del model
    torch.cuda.empty_cache()
    return question_to_response


def load_triviaqa_dataset() -> List[Dict]:
    """Load and sample TriviaQA dataset."""
    print(f"Loading TriviaQA dataset (unfiltered.nocontext)...")
    dataset = load_dataset("trivia_qa", "unfiltered.nocontext", split="validation")
    
    
    # Extract questions and answer aliases
    inputs = []
    for item in dataset:
        inputs.append({
            "id": item.get("id", ""),
            "question": item.get("question", ""),
            "answer": item.get("answer", {}),
        })
    
    return inputs


def evaluate_triviaqa(model_name: str, device: str, inputs: List[Dict]) -> Tuple[List[Dict], Dict[str, float]]:
    """Evaluate a model on TriviaQA."""
    print(f"Evaluating TriviaQA for {model_name}...")
    
    # Generate responses
    question_to_response = generate_responses_for_model(model_name, device, inputs)
    
    # Compute metrics for each example
    results = []
    for inp in tqdm(inputs, desc="Computing metrics", leave=False):
        question = inp["question"]
        answer_obj = inp["answer"]
        
        # Get gold aliases
        gold_aliases = answer_obj.get("aliases", [])
        if not gold_aliases:
            # Fallback to value
            gold_aliases = [answer_obj.get("value", "")]
        
        # Get prediction
        prediction = question_to_response.get(question, "")
        
        # Compute metrics
        metrics = compute_metrics(prediction, gold_aliases)
        
        # Store result
        result = {
            "id": inp["id"],
            "question": question,
            "gold": gold_aliases,
            "prediction": prediction,
            "em": metrics["em"],
            "f1": metrics["f1"],
            "recall": metrics["recall"],
        }
        results.append(result)
    
    # Compute aggregate metrics
    if results:
        agg_metrics = {
            "em": sum(r["em"] for r in results) / len(results),
            "f1": sum(r["f1"] for r in results) / len(results),
            "recall": sum(r["recall"] for r in results) / len(results),
        }
    else:
        agg_metrics = {"em": 0.0, "f1": 0.0, "recall": 0.0}
    
    return results, agg_metrics


def main():
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= N_GPUS, f"Require at least {N_GPUS} GPUs. Found {n_gpus}."
    
    # Load dataset
    inputs = load_triviaqa_dataset()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # Evaluate each model
    aggregate = {}
    for mi, mname in enumerate(tqdm(MODEL_NAMES, desc="Models", leave=True)):
        device = f"cuda:{mi % N_GPUS}" if torch.cuda.is_available() else "cpu"
        
        # Evaluate
        results, metrics = evaluate_triviaqa(mname, device, inputs)
        
        # Save per-example results
        model_tag = mname.split("/")[-1]
        out_dir = os.path.join(OUTPUT_ROOT, model_tag)
        os.makedirs(out_dir, exist_ok=True)
        results_path = os.path.join(out_dir, "results.jsonl")
        with open(results_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        
        # Store aggregate metrics
        aggregate[mname] = metrics
    
    # Write metrics summary
    metrics_path = os.path.join(OUTPUT_ROOT, "triviaqa_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("===== TriviaQA Metrics =====\n")
        for m, s in aggregate.items():
            f.write(f"{m}: EM={s['em']:.4f}, F1={s['f1']:.4f}, Recall={s['recall']:.4f}\n")
    
    print(f"\nTriviaQA metrics written to {metrics_path}")
    
    # Also save as JSON
    metrics_json_path = os.path.join(OUTPUT_ROOT, "triviaqa_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"TriviaQA metrics (JSON) written to {metrics_json_path}")


if __name__ == "__main__":
    main()

