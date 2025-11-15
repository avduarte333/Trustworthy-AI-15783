import os
import sys
import json
from typing import List, Dict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import nltk
nltk.data.path.append("/home/aduarte/nltk_data")

# Configuration
MODEL_NAMES = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
]
N_GPUS = 4
BATCH_SIZE = 8
MAX_NEW_TOKENS = 1024

IFEVAL_DIR = "/home/aduarte/Trustworthy_AI/HW2/IFEval"
INPUT_JSONL = os.path.join(IFEVAL_DIR, "data", "input_data.jsonl")
OUTPUT_ROOT = "/home/aduarte/Trustworthy_AI/HW2/logs/ifeval"


def read_prompt_list(path: str) -> List[Dict]:
    prompts = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompts.append(obj)
    return prompts


def build_chat_prompt(tokenizer, user_prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful assistant. Follow the user's instructions exactly. "
                "Do not add disclaimers or extra commentary beyond what is requested."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate_responses_for_model(model_name: str, device: str, inputs: List[Dict]) -> Dict[str, str]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": device},
        torch_dtype="auto",
    )
    model.eval()

    prompt_texts = [ex["prompt"] for ex in inputs]
    prompt_to_response: Dict[str, str] = {}

    total_batches = (len(prompt_texts) + BATCH_SIZE - 1) // BATCH_SIZE if prompt_texts else 0
    for i in tqdm(range(0, len(prompt_texts), BATCH_SIZE), total=total_batches, desc=f"{model_name} batches", leave=False):
        batch_prompts = prompt_texts[i : i + BATCH_SIZE]
        chat_batch = [build_chat_prompt(tokenizer, p) for p in batch_prompts]
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
        for idx, p in enumerate(batch_prompts):
            prompt_len = len(inputs_enc["input_ids"][idx])
            out_ids = outputs[idx][prompt_len:]
            text = tokenizer.decode(out_ids, skip_special_tokens=True).strip()
            prompt_to_response[p] = text

    # Cleanup
    del model
    torch.cuda.empty_cache()
    return prompt_to_response


def write_prompt_response_jsonl(path: str, prompt_to_response: Dict[str, str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for prompt, response in prompt_to_response.items():
            f.write(json.dumps({"prompt": prompt, "response": response}))
            f.write("\n")


def run_ifeval_scoring(input_jsonl: str, response_jsonl: str, out_dir: str):
    # Add IFEVAL_DIR to sys.path so we can import the modules directly
    if IFEVAL_DIR not in sys.path:
        sys.path.insert(0, IFEVAL_DIR)
    
    # Create alias package so imports like `from instruction_following_eval import X` work
    import types
    pkg = types.ModuleType("instruction_following_eval")
    pkg.__path__ = [IFEVAL_DIR]  # type: ignore[attr-defined]
    sys.modules["instruction_following_eval"] = pkg
    
    # Import modules in dependency order and register them immediately
    # 1. instructions_util (no dependencies on other modules)
    import instructions_util
    pkg.instructions_util = instructions_util
    sys.modules["instruction_following_eval.instructions_util"] = instructions_util
    
    # 2. instructions (depends on instructions_util)
    import instructions
    pkg.instructions = instructions
    sys.modules["instruction_following_eval.instructions"] = instructions
    
    # 3. instructions_registry (depends on instructions)
    import instructions_registry
    pkg.instructions_registry = instructions_registry
    sys.modules["instruction_following_eval.instructions_registry"] = instructions_registry
    
    # 4. evaluation_lib (depends on instructions_registry)
    import evaluation_lib
    pkg.evaluation_lib = evaluation_lib
    sys.modules["instruction_following_eval.evaluation_lib"] = evaluation_lib

    inputs = evaluation_lib.read_prompt_list(input_jsonl)
    prompt_to_response = evaluation_lib.read_prompt_to_response_dict(response_jsonl)

    results = {}
    for func, name in [
        (evaluation_lib.test_instruction_following_strict, "eval_results_strict"),
        (evaluation_lib.test_instruction_following_loose, "eval_results_loose"),
    ]:
        outputs = [func(inp, prompt_to_response) for inp in inputs]
        out_path = os.path.join(out_dir, name + ".jsonl")
        evaluation_lib.write_outputs(out_path, outputs)

        # Compute accuracies
        follow_all = [o.follow_all_instructions for o in outputs]
        accuracy = sum(follow_all) / len(outputs) if outputs else 0.0
        results[name] = accuracy

    return results


def main():
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= N_GPUS, f"Require at least {N_GPUS} GPUs. Found {n_gpus}."

    inputs = read_prompt_list(INPUT_JSONL)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)


    aggregate = {}
    for mi, mname in enumerate(tqdm(MODEL_NAMES, desc="Models", leave=True)):
        print(f"\nEvaluating IFEval for {mname}...")
        device = f"cuda:{mi % N_GPUS}" if torch.cuda.is_available() else "cpu"

        # Inference
        prompt_to_response = generate_responses_for_model(mname, device, inputs)

        # Persist responses
        model_tag = mname.split("/")[-1]
        out_dir = os.path.join(OUTPUT_ROOT, model_tag)
        os.makedirs(out_dir, exist_ok=True)
        response_path = os.path.join(out_dir, "responses.jsonl")
        write_prompt_response_jsonl(response_path, prompt_to_response)

        # Score with IFEval
        scores = run_ifeval_scoring(INPUT_JSONL, response_path, out_dir)
        aggregate[mname] = scores

    # Write a compact metrics file
    metrics_path = os.path.join(OUTPUT_ROOT, "ifeval_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("===== IFEval Accuracies (prompt-level follow_all) =====\n")
        for m, s in aggregate.items():
            f.write(f"{m}: strict={s['eval_results_strict']:.4f}, loose={s['eval_results_loose']:.4f}\n")
    print(f"\nIFEval metrics written to {metrics_path}")


if __name__ == "__main__":
    main()

