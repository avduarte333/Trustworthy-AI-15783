import json
import os
import torch
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils_mcqa import load_mcqa, build_prompt_chat, build_prompt_freeform, extract_choice, check_free_form_answer
from tqdm import tqdm

@torch.no_grad()
def mcqa_accuracy(model, tokenizer, items, device=None, batch_size=16, experiment_type="mcqa"):
    """Compute per-trait accuracy with batch processing. Supports both MCQA and free-form."""
    model.eval()
    by_trait = defaultdict(lambda: [0, 0])

    # Process in batches for efficiency
    for i in tqdm(range(0, len(items), batch_size), desc="Evaluating", leave=False):
        batch = items[i:i+batch_size]
        
        # Build prompts based on experiment type
        if experiment_type == "mcqa":
            prompts = [build_prompt_chat(tokenizer, ex.question, ex.options) for ex in batch]
            max_tokens = 8
        else:  # free-form
            prompts = [build_prompt_freeform(tokenizer, ex.question) for ex in batch]
            max_tokens = 50
        
        # With device_map="auto", inputs will be moved automatically
        inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        
        for idx, ex in enumerate(batch):
            prompt_len = len(inputs['input_ids'][idx])
            output_ids = outputs[idx][prompt_len:]
            response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            
            by_trait[ex.trait][1] += 1
            
            # Check correctness based on experiment type
            if experiment_type == "mcqa":
                pred = extract_choice(response)
                if pred is not None and pred in ex.answer:
                    by_trait[ex.trait][0] += 1
            else:  # free-form
                if ex.answer_full_writing and check_free_form_answer(response, ex.answer_full_writing):
                    by_trait[ex.trait][0] += 1

    return {t: (c / tot if tot > 0 else 0.0) for t, (c, tot) in by_trait.items()}

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate unlearned models")
    parser.add_argument(
        "--experiment_type",
        type=str,
        default="mcqa",
        choices=["mcqa", "free-form"],
        help="Type of experiment: 'mcqa' for multiple choice or 'free-form' for free-form generation"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["base", "ga", "dpo"],
        choices=["base", "ga", "dpo"],
        help="Which models to evaluate: choose from base ga dpo (space-separated)"
    )
    args = parser.parse_args()

    experiment_type = args.experiment_type
    selected_models = args.models
    print(f"Running evaluation with experiment type: {experiment_type}")
    print(f"Models selected for evaluation: {', '.join(selected_models)}")

    model_name = "Qwen/Qwen3-4B"
    held_out_trait = "type1"
    all_traits = ["type1", "hp", "speed", "defense"]
    forget_traits = [t for t in all_traits if t != held_out_trait]
    items = load_mcqa("/home/aduarte/Trustworthy_AI/HW2/pokemon_mcqa.json")
    batch_size = 16

    # Get HuggingFace cache directory
    hf_home = os.environ.get("HF_HOME", "/data/user_data/aduarte/HuggingFace")

    # Build model paths only for selected labels
    all_paths = {
        "base": model_name,  # remote model id
        "ga": os.path.join(hf_home, f"{model_name}-unlearned-ga-{experiment_type}"),
        "dpo": os.path.join(hf_home, f"{model_name}-unlearned-dpo-{experiment_type}"),
    }
    paths = {k: all_paths[k] for k in selected_models}

    # Determine device - use multiple GPUs if available
    n_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {n_gpus}")

    print("\nEvaluating selected models...")
    results = {}
    actually_evaluated = []
    for label, path in tqdm(paths.items(), desc="Models"):
        # For finetuned models (ga/dpo), ensure the directory exists; for base, allow remote
        if label != "base" and not os.path.isdir(path):
            print(f" -> Skipping {label}: directory not found at {path}")
            continue

        print(f"\n -> Evaluating {label}: {path}")
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="auto",
            dtype="auto"
        )

        print(f"    Model device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'single device'}")

        results[label] = mcqa_accuracy(model, tokenizer, items, device=None, batch_size=batch_size, experiment_type=experiment_type)
        actually_evaluated.append(label)

        # Print per-trait results
        print(f"    Results for {label}:")
        for trait in all_traits:
            acc = results[label].get(trait, 0.0)
            print(f"      {trait}: {acc:.4f}")

        # Clean up memory
        del model
        torch.cuda.empty_cache()

    if not actually_evaluated:
        print("No models were evaluated. Check your --models selection and that finetuned model directories exist.")
        return

    def avg(d, traits):
        return sum(d.get(t, 0) for t in traits) / len(traits)

    # Summary only for evaluated labels
    labels = actually_evaluated
    forget = [avg(results[k], forget_traits) for k in labels]
    held = [results[k].get(held_out_trait, 0) for k in labels]

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY RESULTS")
    print("=" * 60)
    print(f"{'Model':<10} {'Forget Traits Avg':<20} {'Held-out (type1)':<20}")
    print("-" * 60)
    for i, model_label in enumerate(labels):
        print(f"{model_label.upper():<10} {forget[i]:>18.4f}   {held[i]:>18.4f}")
    print("=" * 60)

    # --- Grouped Bar Chart --- (dynamic number of bars)
    plt.figure(figsize=(8, 5))
    x = range(len(labels))
    width = 0.35
    plt.bar([i - width / 2 for i in x], forget, width, label="Forget Traits")
    plt.bar([i + width / 2 for i in x], held, width, label="Held-out Trait")
    plt.xticks(list(x), [lbl.upper() for lbl in labels])
    plt.ylabel("Accuracy")
    plt.title(f"Part 2 – Targeted Unlearning (Qwen3-4B, {experiment_type})")
    plt.legend()
    plt.tight_layout()
    tag = "-".join(labels)
    fig_name = f"part2_grouped_bar_{experiment_type}_{tag}.png"
    plt.savefig(fig_name, dpi=200)
    print(f"\nSaved figure: {fig_name}")

    results_name = f"part2_results_{experiment_type}_{tag}.json"
    with open(results_name, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_name}")

if __name__ == "__main__":
    main()
