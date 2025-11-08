import os
import math
import json
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from tqdm import tqdm

from utils_mcqa import load_mcqa, build_prompt_chat, build_prompt_freeform, extract_choice

REFUSAL = "I cannot provide this information."

class DPODataset(Dataset):
    def __init__(self, items, held_out_trait, tokenizer, experiment_type="mcqa", max_len=512):
        # Create pairs: forget examples (prefer refusal) + retain examples (prefer correct answer)
        forget_examples = [ex for ex in items if ex.trait != held_out_trait]
        retain_examples = [ex for ex in items if ex.trait == held_out_trait]
        
        # Balance dataset: use all forget examples + sample retain examples to match
        # Use 50% retain examples to prevent collapse while still emphasizing unlearning
        num_retain = min(len(retain_examples), len(forget_examples) // 2)
        import random
        random.seed(42)
        retain_examples = random.sample(retain_examples, num_retain)
        
        # Mark each example as forget or retain
        self.rows = [(ex, "forget") for ex in forget_examples] + [(ex, "retain") for ex in retain_examples]
        random.shuffle(self.rows)
        
        self.tokenizer = tokenizer
        self.experiment_type = experiment_type
        self.max_len = max_len
        
        print(f"Dataset: {len(forget_examples)} forget + {len(retain_examples)} retain = {len(self.rows)} total")

    def __len__(self): 
        return len(self.rows)

    def __getitem__(self, idx):
        ex, example_type = self.rows[idx]
        
        # Build prompt based on experiment type
        if self.experiment_type == "mcqa":
            prompt = build_prompt_chat(self.tokenizer, ex.question, ex.options)
            correct_answer = extract_choice(ex.answer)
            if correct_answer is None:
                correct_answer = ex.answer[0] if ex.answer[0] in ["A", "B", "C", "D"] else "A"
        else:  # free-form
            prompt = build_prompt_freeform(self.tokenizer, ex.question)
            correct_answer = ex.answer_full_writing if ex.answer_full_writing else ex.answer
        
        # For forget examples: prefer refusal over correct answer
        # For retain examples: prefer correct answer over refusal (REVERSED)
        if example_type == "forget":
            chosen = REFUSAL
            rejected = correct_answer
        else:  # retain
            chosen = correct_answer
            rejected = REFUSAL
        
        enc_prompt = self.tokenizer(prompt, return_tensors="pt",
                                    truncation=True, max_length=self.max_len)
        enc_chosen = self.tokenizer(chosen, return_tensors="pt", add_special_tokens=False)
        enc_rejected = self.tokenizer(rejected, return_tensors="pt", add_special_tokens=False)
        return {
            "prompt_ids": enc_prompt["input_ids"][0],
            "prompt_am":  enc_prompt["attention_mask"][0],
            "chosen_ids": enc_chosen["input_ids"][0],
            "rejected_ids": enc_rejected["input_ids"][0],
        }

def collate_pad(batch, pad_id):
    def pad_to_max(tensors, pad_val):
        L = max(t.size(0) for t in tensors)
        out = []
        for t in tensors:
            if t.size(0) < L:
                t = torch.cat([t, torch.full((L - t.size(0),), pad_val, dtype=t.dtype)])
            out.append(t)
        return torch.stack(out, 0)

    return {
        "prompt_ids": pad_to_max([b["prompt_ids"] for b in batch], pad_id),
        "prompt_am":  pad_to_max([b["prompt_am"] for b in batch], 0),
        "chosen_ids": pad_to_max([b["chosen_ids"] for b in batch], -100),
        "rejected_ids": pad_to_max([b["rejected_ids"] for b in batch], -100),
    }

def seq_logprob(model, input_ids, attention_mask, resp_ids):
    """Compute log p(response | prompt) with batch padding for variable-length sequences."""
    concat_ids, concat_am, labels = [], [], []
    for i in range(input_ids.size(0)):
        # Filter out padding tokens (-100) from response
        valid_resp = resp_ids[i][resp_ids[i] != -100]
        if valid_resp.size(0) == 0:
            # Skip if no valid response tokens
            continue
        full = torch.cat([input_ids[i], valid_resp])
        am   = torch.cat([attention_mask[i], torch.ones_like(valid_resp)])
        lab  = torch.full_like(full, -100)
        lab[-valid_resp.size(0):] = valid_resp
        concat_ids.append(full); concat_am.append(am); labels.append(lab)

    if len(concat_ids) == 0:
        # Return zero log probability if no valid examples
        return torch.zeros(input_ids.size(0), device=input_ids.device)

    # Pad to the max length in the batch before stacking
    def pad_to_max(seqs, pad_val):
        L = max(t.size(0) for t in seqs)
        out = []
        for t in seqs:
            if t.size(0) < L:
                t = torch.cat([
                    t,
                    torch.full((L - t.size(0),), pad_val, dtype=t.dtype, device=t.device)
                ])
            out.append(t)
        return torch.stack(out, 0)

    pad_id = getattr(model.config, "pad_token_id", None)
    if pad_id is None:
        pad_id = 0
    concat_ids = pad_to_max(concat_ids, pad_id)
    concat_am = pad_to_max(concat_am, 0)
    labels = pad_to_max(labels, -100)

    out = model(input_ids=concat_ids, attention_mask=concat_am, labels=labels)

    # Compute per-sequence log probabilities from logits
    logits = out.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    flat_loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    token_loss = flat_loss.view(shift_labels.size(0), -1)
    
    # Use TOTAL log probability (sum over tokens), not averaged
    # This is crucial for DPO with very different sequence lengths:
    # - Short answer "grass" (1 token): log prob ≈ -2.0
    # - Long refusal (8 tokens): log prob ≈ -16.0
    # DPO needs this difference to learn to prefer the refusal despite it being longer
    seq_loss_sum = token_loss.sum(dim=1)
    seq_logp = -seq_loss_sum  # Total log probability

    return seq_logp

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train DPO unlearning")
    parser.add_argument(
        "--experiment_type",
        type=str,
        default="mcqa",
        choices=["mcqa", "free-form"],
        help="Type of experiment: 'mcqa' for multiple choice or 'free-form' for free-form generation"
    )
    args = parser.parse_args()
    
    experiment_type = args.experiment_type
    
    model_name = "Qwen/Qwen3-4B"
    held_out_trait = "type1"
    data_path = "/home/aduarte/Trustworthy_AI/HW2/pokemon_mcqa.json"
    
    # Save to HuggingFace cache directory with experiment type
    hf_home = os.environ.get("HF_HOME", "/data/user_data/aduarte/HuggingFace")
    output_dir = os.path.join(hf_home, f"{model_name}-unlearned-dpo-{experiment_type}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Training hyperparameters
    num_epochs = 3
    batch_size = 8
    # Adjust learning rate and beta based on experiment type
    if experiment_type == "free-form":
        learning_rate = 1e-5  # Moderate LR to prevent collapse
        beta = 0.1  # Lower beta for more stable training with total log probs
    else:
        learning_rate = 5e-6
        beta = 0.1
    
    print(f"Starting DPO unlearning")
    print(f"Experiment type: {experiment_type}")
    print(f"Model: {model_name}")
    print(f"Held-out trait: {held_out_trait}")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}, LR: {learning_rate}, Beta: {beta}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    policy = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically split model across GPUs
        dtype="auto"
    )
    reference = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically split model across GPUs
        dtype="auto"
    )
    # Ensure models know the pad token id
    if getattr(policy.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        policy.config.pad_token_id = tokenizer.pad_token_id
    if getattr(reference.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        reference.config.pad_token_id = tokenizer.pad_token_id
    reference.eval()
    for p in reference.parameters():
        p.requires_grad = False

    # Check which devices the models are on
    print(f"Policy model device map: {policy.hf_device_map if hasattr(policy, 'hf_device_map') else 'single device'}")
    print(f"Reference model device map: {reference.hf_device_map if hasattr(reference, 'hf_device_map') else 'single device'}")
    policy.train()

    items = load_mcqa(data_path)
    ds = DPODataset(items, held_out_trait, tokenizer, experiment_type=experiment_type)
    
    # Validate data
    print(f"\n[Data Validation] Sample examples:")
    for i in [0, len(ds)//2]:  # Show one forget and one retain example
        sample_item, sample_type = ds.rows[i]
        sample_data = ds[i]
        print(f"\n  Example {i+1} ({sample_type}):")
        print(f"    Question: {sample_item.question[:80]}...")
        print(f"    Trait: {sample_item.trait}")
        if experiment_type == "free-form":
            print(f"    Answer: {sample_item.answer_full_writing[:50] if sample_item.answer_full_writing else 'EMPTY!'}...")
        else:
            print(f"    Answer: {sample_item.answer}")
        print(f"    Chosen: {tokenizer.decode(sample_data['chosen_ids'], skip_special_tokens=True)[:50]}...")
        print(f"    Rejected: {tokenizer.decode(sample_data['rejected_ids'], skip_special_tokens=True)[:50]}...")
    print()
    
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    collate_fn=lambda b: collate_pad(b, tokenizer.pad_token_id))

    optim = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
    total_steps = len(dl) * num_epochs
    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=int(total_steps * 0.03),
        num_training_steps=total_steps,
    )

    # Training logs
    training_logs = []
    global_step = 0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        epoch_losses = []
        progress_bar = tqdm(dl, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar, 1):
            global_step += 1
            # With device_map="auto", inputs will be moved to appropriate devices automatically
            prompt_ids = batch["prompt_ids"].cuda()
            prompt_am = batch["prompt_am"].cuda()
            chosen_ids = batch["chosen_ids"].cuda()
            rejected_ids = batch["rejected_ids"].cuda()

            # Compute log probabilities
            pi_c = seq_logprob(policy, prompt_ids, prompt_am, chosen_ids)
            pi_r = seq_logprob(policy, prompt_ids, prompt_am, rejected_ids)
            
            with torch.no_grad():
                ref_c = seq_logprob(reference, prompt_ids, prompt_am, chosen_ids)
                ref_r = seq_logprob(reference, prompt_ids, prompt_am, rejected_ids)

            # DPO loss
            # We want to maximize: log(sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r))))
            # Which is equivalent to minimizing: softplus(-beta * margin)
            # Positive margin means policy prefers chosen (refusal) over rejected (correct answer)
            d = beta * ((pi_c - ref_c) - (pi_r - ref_r))
            loss = torch.nn.functional.softplus(-d).mean()
            
            # Track individual components for debugging
            if global_step == 1 or global_step % 20 == 0:
                with torch.no_grad():
                    # Calculate implicit rewards
                    reward_chosen = beta * (pi_c - ref_c)
                    reward_rejected = beta * (pi_r - ref_r)
                    margin = d / beta
                    
                print(f"\n[Step {global_step}] Debug info:")
                print(f"  Policy log probs:")
                print(f"    pi_c (refusal):  {pi_c.mean().item():.4f} (total, not per-token)")
                print(f"    pi_r (correct):  {pi_r.mean().item():.4f} (total, not per-token)")
                print(f"  Reference log probs:")
                print(f"    ref_c: {ref_c.mean().item():.4f}")
                print(f"    ref_r: {ref_r.mean().item():.4f}")
                print(f"  Log ratios (policy - reference):")
                print(f"    chosen:  {(pi_c - ref_c).mean().item():.4f}")
                print(f"    rejected: {(pi_r - ref_r).mean().item():.4f}")
                print(f"  Margin (chosen_ratio - rejected_ratio): {margin.mean().item():.4f}")
                print(f"    -> Positive = policy prefers refusal (GOOD)")
                print(f"    -> Negative = policy prefers correct answer (BAD)")
                print(f"  Loss: {loss.item():.4f}")
                print(f"  Learning rate: {sched.get_last_lr()[0]:.2e}")

            optim.zero_grad()
            loss.backward()
            # Add gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optim.step()
            sched.step()
            
            epoch_losses.append(loss.item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'margin': f'{d.mean().item():.4f}',
                'lr': f'{sched.get_last_lr()[0]:.2e}'
            })
            
            # Log periodically
            if global_step % 20 == 0:
                log_entry = {
                    'epoch': epoch + 1,
                    'step': step,
                    'global_step': global_step,
                    'loss': loss.item(),
                    'margin': d.mean().item(),
                    'lr': sched.get_last_lr()[0]
                }
                training_logs.append(log_entry)
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1} completed - Average Loss: {avg_loss:.4f}")
        print(f"{'='*60}")
        
        # Quick validation: test a few examples to see if unlearning is happening
        print("\n--- Quick Validation Sample ---")
        policy.eval()
        with torch.no_grad():
            # Test forget examples (should refuse)
            forget_examples = [ex for ex, typ in ds.rows if typ == "forget"][:2]
            # Test retain examples (should answer)
            retain_examples = [ex for ex, typ in ds.rows if typ == "retain"][:2]
            
            print("FORGET trait examples (should refuse):")
            for ex in forget_examples:
                if experiment_type == "free-form":
                    prompt = build_prompt_freeform(tokenizer, ex.question)
                else:
                    prompt = build_prompt_chat(tokenizer, ex.question, ex.options)
                
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                outputs = policy.generate(**inputs, max_new_tokens=50, do_sample=False)
                response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
                
                correct = ex.answer_full_writing if experiment_type == 'free-form' else ex.answer
                print(f"  Q: {ex.question[:60]}...")
                print(f"  Trait: {ex.trait}, Correct: {correct[:30]}")
                print(f"  Model: {response[:80]}")
                
            print("\nRETAIN trait examples (should answer correctly):")
            for ex in retain_examples:
                if experiment_type == "free-form":
                    prompt = build_prompt_freeform(tokenizer, ex.question)
                else:
                    prompt = build_prompt_chat(tokenizer, ex.question, ex.options)
                
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                outputs = policy.generate(**inputs, max_new_tokens=50, do_sample=False)
                response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
                
                correct = ex.answer_full_writing if experiment_type == 'free-form' else ex.answer
                print(f"  Q: {ex.question[:60]}...")
                print(f"  Trait: {ex.trait}, Correct: {correct[:30]}")
                print(f"  Model: {response[:80]}")
        policy.train()
        print("--- End Validation ---\n")
        
        # Save checkpoint after each epoch
        checkpoint_dir = os.path.join(hf_home, f"{model_name}-unlearned-dpo-{experiment_type}-epoch{epoch+1}")
        policy.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"Checkpoint saved to {checkpoint_dir}")

    # Save final model
    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nFinal model saved to {output_dir}")
    
    # Save training logs
    with open(os.path.join(output_dir, "training_logs.json"), "w") as f:
        json.dump(training_logs, f, indent=2)
    print(f"Training logs saved to {os.path.join(output_dir, 'training_logs.json')}")
    
    # Clean up
    del policy, reference
    torch.cuda.empty_cache()
    print("\nTraining completed!")

if __name__ == "__main__":
    main()
