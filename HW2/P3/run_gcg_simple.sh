#!/bin/bash
#SBATCH --job-name=gcg_simple
#SBATCH --partition=general
#SBATCH --gres=gpu:6000Ada:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out

# Simple GCG Attack - Two Steps:
# 1. Find the adversarial suffix
# 2. Evaluate with that suffix

set -euo pipefail

echo "=========================================="
echo "Simple GCG Attack"
echo "=========================================="
echo "Start time: $(date)"

# Load environment
source ~/miniconda3/bin/activate cmu

# Move to project directory
cd /home/aduarte/Trustworthy_AI/HW2/P3

# Config
HF_HOME="${HF_HOME:-/data/user_data/aduarte/HuggingFace}"
MODEL_PATH="${HF_HOME}/Qwen/Qwen3-4B-unlearned-dpo-free-form"
BENCHMARK_PATH="/home/aduarte/Trustworthy_AI/HW2/pokemon.json"
SUFFIX_FILE="gcg_suffix.txt"

echo "Model: ${MODEL_PATH}"
echo "Benchmark: ${BENCHMARK_PATH}"
echo ""

# Step 1: Find suffix
echo "=========================================="
echo "STEP 1: Finding GCG Suffix"
echo "=========================================="
python gcg_find_suffix.py \
    --model_path "${MODEL_PATH}" \
    --benchmark_path "${BENCHMARK_PATH}" \
    --output_suffix_file "${SUFFIX_FILE}" \
    --num_steps 100 \
    --suffix_length 20 \
    --target "Sure, the" \
    --seed 42

echo ""
echo "✓ Suffix found and saved to ${SUFFIX_FILE}"
echo ""

# Step 2: Evaluate with suffix
echo "=========================================="
echo "STEP 2: Evaluating with Suffix"
echo "=========================================="
python gcg_evaluate.py \
    --model_path "${MODEL_PATH}" \
    --benchmark_path "${BENCHMARK_PATH}" \
    --suffix_file "${SUFFIX_FILE}" \
    --experiment_type free-form \
    --batch_size 16

echo ""
echo "=========================================="
echo "Complete!"
echo "End time: $(date)"
echo "=========================================="