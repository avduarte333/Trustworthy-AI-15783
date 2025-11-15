#!/bin/bash
#SBATCH --job-name=pokemon_part_2
#SBATCH --partition=general
#SBATCH --gres=gpu:6000Ada:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out

# Simple, comment-driven pipeline.
# Comment out any block you don't want to run.

set -euo pipefail

# ---- CONFIG ----
EXPERIMENT_TYPE="free-form"   # mcqa | free-form
EVAL_MODELS="base ga dpo"     # which models to evaluate (space-separated): base ga dpo

echo "=========================================="
echo "Part 2: Targeted Unlearning"
echo "=========================================="
echo "Start time: $(date)"
echo "Experiment type: ${EXPERIMENT_TYPE}"
echo "Evaluate models: ${EVAL_MODELS}"
echo ""

# Load environment
source ~/miniconda3/bin/activate cmu

# Move to project directory
cd /home/aduarte/Trustworthy_AI/HW2/P2

# ==========================================
# Train GA (Gradient Ascent)
#   - Comment this block to skip GA training
# ==========================================
echo "[GA] Training start: $(date)"
python train_ga.py --experiment_type "${EXPERIMENT_TYPE}"
echo "[GA] Training done:  $(date)"
echo ""

# ==========================================
# Train DPO
#   - Comment this block to skip DPO training
# ==========================================
echo "[DPO] Training start: $(date)"
python train_dpo.py --experiment_type "${EXPERIMENT_TYPE}"
echo "[DPO] Training done:  $(date)"
echo ""

# ==========================================
# Evaluate selected models
#   - Comment this block to skip evaluation
# ==========================================
echo "[EVAL] Start: $(date)"
python evaluate_part2.py --experiment_type "${EXPERIMENT_TYPE}" --models ${EVAL_MODELS}
echo "[EVAL] Done:  $(date)"
echo ""

echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "End time: $(date)"
