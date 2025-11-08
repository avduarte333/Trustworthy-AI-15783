"""
Simple GCG suffix optimization.
Finds an adversarial suffix to append to Pokemon questions.
Uses incremental optimization: only moves to next question when current one succeeds.
"""

import json
import torch
import torch.nn.functional as F
import argparse
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def test_suffix_success(model, tokenizer, question, suffix_ids, target_str="Sure, ", verbose=False):
    """Test if the suffix makes the model generate the target string."""
    # Build prompt with suffix
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions directly and concisely."},
        {"role": "user", "content": question}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    
    # Tokenize prompt
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    # Append suffix
    full_ids = torch.cat([prompt_ids, suffix_ids.unsqueeze(0)], dim=1)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(full_ids, max_new_tokens=35, do_sample=False)
    
    # Decode response
    response_ids = outputs[0][full_ids.shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    print(response)
    
    success = response.startswith(target_str)
    
    if verbose:
        print(f"    Response: '{response[:50]}...' | Success: {success}")
    
    return success


def compute_grad_and_loss(model, tokenizer, question, suffix_ids, target_str, embedding_layer):
    """Compute gradient and loss for one question."""
    vocab_size = embedding_layer.num_embeddings
    device = next(model.parameters()).device
    
    # Build prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    
    # Tokenize components
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt')[0].to(device)
    target_ids = tokenizer.encode(" " + target_str, add_special_tokens=False, return_tensors='pt')[0].to(device)
    
    # Create one-hot for suffix (match model dtype)
    suffix_onehot = F.one_hot(suffix_ids, num_classes=vocab_size)
    suffix_onehot = suffix_onehot.to(embedding_layer.weight.dtype).requires_grad_(True)
    
    # Get embeddings
    prompt_embeds = embedding_layer(prompt_ids)
    suffix_embeds = suffix_onehot @ embedding_layer.weight
    target_embeds = embedding_layer(target_ids)
    
    # Concatenate
    input_embeds = torch.cat([prompt_embeds, suffix_embeds, target_embeds], dim=0).unsqueeze(0)
    
    # Create labels (only compute loss on target)
    labels = torch.full((input_embeds.shape[1],), -100, dtype=torch.long, device=device)
    target_start = prompt_embeds.shape[0] + suffix_embeds.shape[0]
    labels[target_start:target_start + len(target_ids)] = target_ids
    
    # Forward pass
    outputs = model(inputs_embeds=input_embeds, labels=labels.unsqueeze(0))
    loss = outputs.loss
    
    # Backward
    loss.backward()
    
    # Get gradient
    grad = suffix_onehot.grad.clone()
    
    return loss.item(), grad


def optimize_suffix(
    model,
    tokenizer,
    questions,
    target_str="Sure, ",
    suffix_length=20,
    num_steps=500,
    topk=1024,
    batch_size=512
):
    """
    Optimize adversarial suffix using incremental GCG.
    Only moves to next question when current one succeeds.
    """
    embedding_layer = model.get_input_embeddings()
    vocab_size = embedding_layer.num_embeddings
    device = next(model.parameters()).device
    
    # Initialize suffix with "! ! ! ..."
    initial_suffix = " ".join(["!"] * suffix_length)
    suffix_ids = tokenizer.encode(initial_suffix, add_special_tokens=False, return_tensors='pt')[0].to(device)
    
    # Start with first question only
    m_c = 1  # Number of questions we're optimizing over
    m = len(questions)  # Total questions
    
    print(f"Starting optimization with {m} total questions")
    print(f"Target: '{target_str}'")
    print(f"Initial suffix: '{tokenizer.decode(suffix_ids)}'")
    print(f"Will log progress every 10 steps...")
    print(f"{'='*80}\n", flush=True)
    
    for step in range(num_steps):
        # Compute gradients for first m_c questions
        total_grad = None
        total_loss = 0
        
        for j in range(m_c):
            loss, grad = compute_grad_and_loss(
                model, tokenizer, questions[j], suffix_ids, target_str, embedding_layer
            )
            
            total_loss += loss
            if total_grad is None:
                total_grad = grad
            else:
                total_grad += grad
        
        avg_loss = total_loss / m_c
        
        # Sample candidate replacements
        candidates = []
        for _ in range(batch_size):
            candidate = suffix_ids.clone()
            # Pick random position
            pos = random.randint(0, len(suffix_ids) - 1)
            # Get top-k tokens for that position
            pos_grad = total_grad[pos]
            topk_ids = torch.topk(-pos_grad, k=min(topk, vocab_size)).indices
            # Pick random from top-k
            new_token = topk_ids[random.randint(0, min(topk, len(topk_ids)) - 1)]
            candidate[pos] = new_token
            candidates.append(candidate)
        
        # Evaluate candidates
        best_candidate = suffix_ids
        best_candidate_loss = avg_loss
        
        for candidate in candidates:
            # Compute loss for this candidate
            cand_loss = 0
            for j in range(m_c):
                loss, _ = compute_grad_and_loss(
                    model, tokenizer, questions[j], candidate, target_str, embedding_layer
                )
                cand_loss += loss
            cand_loss /= m_c
            
            if cand_loss < best_candidate_loss:
                best_candidate_loss = cand_loss
                best_candidate = candidate
        
        suffix_ids = best_candidate
        
        # Check if we succeed on all m_c questions
        if step % 10 == 0:  # Check every 10 steps
            all_success = True
            for j in range(m_c):
                if not test_suffix_success(model, tokenizer, questions[j], suffix_ids, target_str):
                    all_success = False
                    break
            
            current_suffix = tokenizer.decode(suffix_ids, skip_special_tokens=True)
            print(f"\n{'='*80}", flush=True)
            print(f"Step {step}/{num_steps} | Loss: {avg_loss:.4f} | m_c: {m_c}/{m}", flush=True)
            print(f"Current suffix: '{current_suffix}'", flush=True)
            print(f"Testing first {m_c} question(s):", flush=True)
            
            # Test each question in current set
            for j in range(min(m_c, 3)):  # Show first 3
                q_success = test_suffix_success(model, tokenizer, questions[j], suffix_ids, target_str)
                status = "✓" if q_success else "✗"
                print(f"  Q{j+1}: {status} '{questions[j][:60]}...'", flush=True)
            
            if m_c > 3:
                print(f"  ... and {m_c - 3} more questions", flush=True)
            
            if all_success and m_c < m:
                m_c += 1
                print(f"\n🎉 All {m_c-1} questions succeeded! Incrementing to m_c={m_c}/{m}", flush=True)
            elif all_success:
                print(f"\n🎉 All {m_c} questions succeeded! (Maximum reached)", flush=True)
            else:
                print(f"\n⚠️  Not all questions passed yet. Continuing optimization...", flush=True)
            print(f"{'='*80}\n", flush=True)
        
        # Simple progress indicator every step
        if step % 1 == 0:
            print(f"Step {step+1}/{num_steps} | Loss: {avg_loss:.4f} | m_c: {m_c}/{m}", end='\r', flush=True)
    
    best_suffix_str = tokenizer.decode(suffix_ids, skip_special_tokens=True)
    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Successfully optimized for: {m_c}/{m} questions")
    print(f"Final suffix: '{best_suffix_str}'")
    print(f"\nTesting final suffix on first 5 questions:")
    for j in range(min(5, m)):
        success = test_suffix_success(model, tokenizer, questions[j], suffix_ids, target_str)
        status = "✓" if success else "✗"
        print(f"  Q{j+1}: {status} '{questions[j][:60]}...'")
    print(f"{'='*80}\n")
    
    return best_suffix_str, suffix_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--benchmark_path", type=str, required=True)
    parser.add_argument("--output_suffix_file", type=str, default="gcg_suffix.txt")
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--suffix_length", type=int, default=20)
    parser.add_argument("--target", type=str, default="Sure, the answer is ")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load model
    print(f"\n{'='*80}")
    print("GCG SUFFIX OPTIMIZATION")
    print(f"{'='*80}")
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    print(f"✓ Model loaded")
    
    # Load data - filter for forgotten traits only
    print(f"\nLoading benchmark from {args.benchmark_path}...")
    with open(args.benchmark_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total questions in benchmark: {len(data)}")
    
    # Filter for forgotten traits (not type1, which is held-out/retained)
    forgotten_traits = ["hp", "defense", "speed"]
    print(f"Filtering for forgotten traits: {forgotten_traits}")
    train_data = [item for item in data if item.get("trait", "").lower() in forgotten_traits]
    
    print(f"Questions with forgotten traits: {len(train_data)}")
    
    # Shuffle and limit
    random.shuffle(train_data)
    train_data = train_data[:50]  # Use 50 questions
    train_questions = [item['question'] for item in train_data]
    
    print(f"Using {len(train_questions)} questions")
    print(f"\nExample questions:")
    for i in range(min(3, len(train_questions))):
        print(f"  {i+1}. {train_questions[i]}")
    print(f"{'='*80}\n")
    
    # Optimize suffix
    best_suffix, suffix_ids = optimize_suffix(
        model=model,
        tokenizer=tokenizer,
        questions=train_questions,
        target_str=args.target,
        suffix_length=args.suffix_length,
        num_steps=args.num_steps
    )
    
    # Save suffix (both string and token IDs)
    with open(args.output_suffix_file, 'w') as f:
        f.write(best_suffix)
    
    # Also save token IDs for exact reproduction - USE THE ACTUAL OPTIMIZED TENSOR
    suffix_ids_file = args.output_suffix_file.replace('.txt', '_ids.json')
    with open(suffix_ids_file, 'w') as f:
        # Save the actual optimized token IDs (not re-encoded!)
        suffix_ids_list = suffix_ids.cpu().tolist()
        json.dump({"suffix_ids": suffix_ids_list, "suffix_str": best_suffix}, f, indent=2)
    
    print(f"✓ Suffix saved to {args.output_suffix_file}")
    print(f"✓ Token IDs saved to {suffix_ids_file}")
    print(f"\nNext step: Run evaluation with this suffix:")
    print(f"  python gcg_evaluate.py --suffix_file {args.output_suffix_file} \\")
    print(f"    --model_path {args.model_path} \\")
    print(f"    --benchmark_path {args.benchmark_path}")
    print()


if __name__ == "__main__":
    main()
