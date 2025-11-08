import os
import math
import json
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from tqdm import tqdm

from utils_mcqa import load_mcqa, build_prompt_chat, build_prompt_freeform, extract_choice

class ForgetDataset(Dataset):
    def __init__(self, items, held_out_trait, tokenizer, experiment_type="mcqa", max_len=512):
        self.rows = [ex for ex in items if ex.trait != held_out_trait]
        self.tokenizer = tokenizer
        self.experiment_type = experiment_type
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        ex = self.rows[idx]
        
        # Build prompt based on experiment type
        if self.experiment_type == "mcqa":
            prompt = build_prompt_chat(self.tokenizer, ex.question, ex.options)
            # Extract just the letter from answer (e.g., "A) Grass" -> "A)")
            target = extract_choice(ex.answer)
            if target is None:
                # Fallback: just use first character if it's A/B/C/D
                target = ex.answer[0] if ex.answer[0] in ["A", "B", "C", "D"] else "A"
        else:  # free-form
            prompt = build_prompt_freeform(self.tokenizer, ex.question)
            # Use the full writing answer
            target = ex.answer_full_writing if ex.answer_full_writing else ex.answer
        
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=self.max_len, add_special_tokens=True)
        tgt = self.tokenizer(target, return_tensors="pt", add_special_tokens=False)
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": tgt["input_ids"][0],
        }

def collate_pad(batch, pad_id):
    max_src = max(x["input_ids"].shape[0] for x in batch)
    max_tgt = max(x["labels"].shape[0] for x in batch)
    input_ids, attn, labels = [], [], []
    for x in batch:
        ids, am, lab = x["input_ids"], x["attention_mask"], x["labels"]
        pad_len = max_src - ids.shape[0]
        if pad_len > 0:
            ids = torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
            am = torch.cat([am, torch.zeros((pad_len,), dtype=torch.long)])
        input_ids.append(ids)
        attn.append(am)
        pad_t = max_tgt - lab.shape[0]
        if pad_t > 0:
            lab = torch.cat([lab, torch.full((pad_t,), -100, dtype=torch.long)])
        labels.append(lab)
    return {"input_ids": torch.stack(input_ids, 0),
            "attention_mask": torch.stack(attn, 0),
            "labels": torch.stack(labels, 0)}

def compute_letter_logprob(model, tokenizer, input_ids, attention_mask, target_ids):
    """Compute CE loss over concatenated prompt + target with proper batch padding."""
    B = input_ids.size(0)
    device = input_ids.device
    concat_ids, concat_attn, labels = [], [], []

    for i in range(B):
        ids, am, tgt = input_ids[i], attention_mask[i], target_ids[i]
        # Filter out padding tokens (-100) from target
        valid_tgt = tgt[tgt != -100]
        if valid_tgt.size(0) == 0:
            # Skip if no valid target tokens
            continue
        full = torch.cat([ids, valid_tgt])
        full_am = torch.cat([am, torch.ones_like(valid_tgt)])
        lab = torch.full_like(full, -100)
        lab[-valid_tgt.size(0):] = valid_tgt
        concat_ids.append(full)
        concat_attn.append(full_am)
        labels.append(lab)

    if len(concat_ids) == 0:
        # Return zero loss if no valid examples
        return torch.tensor(0.0, device=device, requires_grad=True)

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

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    concat_ids = pad_to_max(concat_ids, pad_id)
    concat_attn = pad_to_max(concat_attn, 0)
    labels = pad_to_max(labels, -100)

    out = model(input_ids=concat_ids, attention_mask=concat_attn, labels=labels)
    return out.loss

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train GA unlearning")
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
    held_out_trait = "type1"      # held-out trait fixed
    data_path = "/home/aduarte/Trustworthy_AI/HW2/pokemon_mcqa.json"
    
    # Save to HuggingFace cache directory with experiment type
    hf_home = os.environ.get("HF_HOME", "/data/user_data/aduarte/HuggingFace")
    output_dir = os.path.join(hf_home, f"{model_name}-unlearned-ga-{experiment_type}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Training hyperparameters
    num_epochs = 3
    batch_size = 16
    learning_rate = 1e-5
    
    print(f"Starting Gradient Ascent unlearning")
    print(f"Experiment type: {experiment_type}")
    print(f"Model: {model_name}")
    print(f"Held-out trait: {held_out_trait}")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}, LR: {learning_rate}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically split model across GPUs
        dtype="auto"
    )

    # Ensure model also knows the pad token id
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Check which devices the model is on
    print(f"Model device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'single device'}")
    model.train()

    items = load_mcqa(data_path)
    ds = ForgetDataset(items, held_out_trait, tokenizer, experiment_type=experiment_type)
    print(f"Training on {len(ds)} forget-trait examples (excluding {held_out_trait})")
    
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    collate_fn=lambda b: collate_pad(b, tokenizer.pad_token_id))

    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(dl) * num_epochs
    sched = get_linear_schedule_with_warmup(optim,
        num_warmup_steps=int(total_steps*0.03), num_training_steps=total_steps)

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
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            target_ids = batch["labels"].cuda()
            
            ce_loss = compute_letter_logprob(model, tokenizer, input_ids, attention_mask, target_ids)
            # If using DataParallel, ce_loss will be a tensor with one value per GPU
            # Take the mean to get a scalar
            if ce_loss.dim() > 0:
                ce_loss = ce_loss.mean()
            
            loss = -ce_loss  # Gradient Ascent
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()
            
            epoch_losses.append(ce_loss.item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'ce_loss': f'{ce_loss.item():.4f}',
                'ga_loss': f'{loss.item():.4f}',
                'lr': f'{sched.get_last_lr()[0]:.2e}'
            })
            
            # Log periodically
            if global_step % 20 == 0:
                log_entry = {
                    'epoch': epoch + 1,
                    'step': step,
                    'global_step': global_step,
                    'ce_loss': ce_loss.item(),
                    'ga_loss': loss.item(),
                    'lr': sched.get_last_lr()[0]
                }
                training_logs.append(log_entry)
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1} completed - Average CE Loss: {avg_loss:.4f}")
        
        # Save checkpoint after each epoch
        checkpoint_dir = os.path.join(hf_home, f"{model_name}-unlearned-ga-{experiment_type}-epoch{epoch+1}")
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"Checkpoint saved to {checkpoint_dir}")
    
    # Save final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nFinal model saved to {output_dir}")
    
    # Save training logs
    with open(os.path.join(output_dir, "training_logs.json"), "w") as f:
        json.dump(training_logs, f, indent=2)
    print(f"Training logs saved to {os.path.join(output_dir, 'training_logs.json')}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    print("\nTraining completed!")

if __name__ == "__main__":
    main()
