"""
Evaluate model with GCG suffix appended to questions.
"""

import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import os
import argparse


def extract_choice(text):
    """Extract answer choice from model response."""
    for c in ["A)", "B)", "C)", "D)"]:
        if c in text:
            return c
    for c in ["A", "B", "C", "D"]:
        if f"{c})" in text:
            return f"{c})"
    return None


def check_free_form_answer(response, correct_answer):
    """Check if free-form response matches correct answer."""
    response_lower = response.strip().lower()
    correct_lower = correct_answer.strip().lower()
    return correct_lower in response_lower


def load_questions(path):
    """Load questions from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate with GCG suffix")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--benchmark_path", type=str, required=True)
    parser.add_argument("--suffix_file", type=str, required=True,
                       help="File containing the GCG suffix (JSON with token IDs)")
    parser.add_argument("--experiment_type", type=str, default="free-form",
                       choices=["mcqa", "free-form"])
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Use batch_size=1 when concatenating token IDs")
    args = parser.parse_args()
    
    # Load suffix token IDs from JSON file
    suffix_ids_file = args.suffix_file.replace('.txt', '_ids.json')
    print(f"Loading suffix token IDs from: {suffix_ids_file}")
    
    with open(suffix_ids_file, 'r') as f:
        suffix_data = json.load(f)
        suffix_ids_list = suffix_data["suffix_ids"]
        suffix_str = suffix_data["suffix_str"]
    
    print(f"Loaded GCG suffix (string): '{suffix_str}'")
    print(f"Loaded GCG suffix (token IDs): {suffix_ids_list}")
    print(f"Experiment type: {args.experiment_type}")
    print(f"Model: {args.model_path}")
    
    # Load model (exactly like part_1-pokemon.py)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map={"": device},
        torch_dtype="auto"
    )
    
    # Convert suffix IDs to tensor
    suffix_ids = torch.tensor([suffix_ids_list], device=device)
    
    # Load questions
    all_questions = load_questions(args.benchmark_path)
    print(f"Loaded {len(all_questions)} questions")
    
    # Group by trait (exactly like part_1-pokemon.py)
    traits = ["type1", "hp", "defense", "speed"]
    questions_by_trait = {t: [] for t in traits}
    for q in all_questions:
        t = q.get("trait", "").lower()
        if t in questions_by_trait:
            questions_by_trait[t].append(q)
    
    # Results
    results = {t: {"correct": 0, "total": 0} for t in traits}
    
    print(f"\n{'='*80}")
    print("Evaluating with GCG suffix...")
    print(f"{'='*80}\n")
    
    for trait in tqdm(traits, desc="Traits"):
        questions = questions_by_trait[trait]
        
        for i in tqdm(range(0, len(questions), args.batch_size), 
                     desc=f"{trait}", leave=False):
            batch = questions[i:i+args.batch_size]
            
            # Process one question at a time when using token ID concatenation
            for q in batch:
                if args.experiment_type == "mcqa":
                    # MCQA: question + options (without suffix in string)
                    content = q["question"].strip() + "\n" + "\n".join(q["options"]) + "\nAnswer:"
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant that answers multiple choice questions. Your answer should be in the format of A), B), C), D). Do not include any other text in your answer."},
                        {"role": "user", "content": content}
                    ]
                else:  # free-form
                    # Free-form: question only (without suffix in string)
                    content = q["question"].strip()
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant that answers questions directly and concisely."},
                        {"role": "user", "content": content}
                    ]
                
                # Apply chat template and tokenize
                chat_str = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                
                # Tokenize the prompt
                prompt_ids = tokenizer.encode(chat_str, return_tensors="pt").to(device)
                
                # Concatenate suffix at TOKEN LEVEL (not string!)
                full_ids = torch.cat([prompt_ids, suffix_ids], dim=1)
                
                # Generate
                max_tokens = 8 if args.experiment_type == "mcqa" else 50
                
                with torch.no_grad():
                    outputs = model.generate(full_ids, max_new_tokens=max_tokens, do_sample=False)
                
                # Decode response (skip the input including suffix)
                response_ids = outputs[0][full_ids.shape[1]:]
                response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
                
                # Check correctness
                is_correct = False
                if args.experiment_type == "mcqa":
                    pred = extract_choice(response)
                    if pred is not None and pred in q["answer"]:
                        is_correct = True
                else:  # free-form
                    if "answer_full_writing" in q:
                        if check_free_form_answer(response, q["answer_full_writing"]):
                            is_correct = True
                
                if is_correct:
                    results[trait]["correct"] += 1
                results[trait]["total"] += 1
                
                # Log every 10 samples
                if results[trait]["total"] % 10 == 0:
                    print(f"\n{'='*80}")
                    print(f"Sample {results[trait]['total']} - Trait: {trait}")
                    print(f"Question: {q['question']}")
                    print(f"Correct: {q.get('answer_full_writing', q.get('answer'))}")
                    print(f"Response: {response}")
                    print(f"Is Correct: {is_correct}")
    
    # Print results (formatted like part_1-pokemon.py)
    print(f"\n{'='*80}")
    print("RESULTS WITH GCG SUFFIX")
    print(f"{'='*80}")
    print(f"Suffix (string): '{suffix_str}'")
    print(f"Suffix (token IDs): {suffix_ids_list}")
    print(f"\n{'='*80}")
    
    overall_correct = 0
    overall_total = 0
    
    # Table header
    print(f"{'Trait':<10s} | {'Accuracy':>10s} | {'Correct/Total'}")
    print("-" * 50)
    
    for trait in traits:
        correct = results[trait]["correct"]
        total = results[trait]["total"]
        acc = correct / total if total > 0 else 0.0
        print(f"{trait:<10s} | {acc*100:>9.2f}% | {correct}/{total}")
        overall_correct += correct
        overall_total += total
    
    overall_acc = overall_correct / overall_total if overall_total > 0 else 0.0
    print("-" * 50)
    print(f"{'OVERALL':<10s} | {overall_acc*100:>9.2f}% | {overall_correct}/{overall_total}")
    print(f"{'='*80}")
    
    # Save results
    output_file = f"gcg_results_{args.experiment_type}.json"
    results_data = {
        "suffix_str": suffix_str,
        "suffix_ids": suffix_ids_list,
        "experiment_type": args.experiment_type,
        "model_path": args.model_path,
        "overall_accuracy": overall_acc,
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "by_trait": {
            trait: {
                "accuracy": results[trait]["correct"] / results[trait]["total"] if results[trait]["total"] > 0 else 0.0,
                "correct": results[trait]["correct"],
                "total": results[trait]["total"]
            }
            for trait in traits
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to {output_file}\n")


if __name__ == "__main__":
    main()
