#!/bin/bash
#SBATCH --job-name=pokemon_part_3
#SBATCH --partition=general
#SBATCH --gres=gpu:6000Ada:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out

# Part 3: Prompt Variation Attack Pipeline
# Comment out any block you don't want to run.

set -euo pipefail

# ---- CONFIG ----
EXPERIMENT_TYPE="free-form"   # mcqa | free-form
MODEL_TO_ATTACK="dpo"         # which model to attack: base | ga | dpo

# Attack strategies to evaluate (space-separated)
# Options: wheel_of_fortune pokemon_song professor_quiz trainer_battle card_game 
#          encyclopedia video_game trivia_night stream_chat debate_club all
STRATEGIES="wheel_of_fortune pokemon_song professor_quiz"

# Optional: limit samples for testing (comment out for full evaluation)
# NUM_SAMPLES="--num_samples 100"
NUM_SAMPLES=""

echo "=========================================="
echo "Part 3: Prompt Variation Attack"
echo "=========================================="
echo "Start time: $(date)"
echo "Experiment type: ${EXPERIMENT_TYPE}"
echo "Target model: ${MODEL_TO_ATTACK}"
echo "Attack strategies: ${STRATEGIES}"
echo ""

# Load environment
source ~/miniconda3/bin/activate cmu

# Move to project directory
cd /home/aduarte/Trustworthy_AI/HW2/P3

# Get HuggingFace cache directory
HF_HOME="${HF_HOME:-/data/user_data/aduarte/HuggingFace}"
BENCHMARK_PATH="/home/aduarte/Trustworthy_AI/HW2/pokemon.json"

# Build model path
if [ "${MODEL_TO_ATTACK}" == "base" ]; then
    MODEL_PATH="Qwen/Qwen3-4B"
else
    MODEL_PATH="${HF_HOME}/Qwen/Qwen3-4B-unlearned-${MODEL_TO_ATTACK}-${EXPERIMENT_TYPE}"
fi

echo "Model path: ${MODEL_PATH}"
echo "Benchmark: ${BENCHMARK_PATH}"

# Create output directory
OUTPUT_DIR="./recovery_results_${MODEL_TO_ATTACK}_${EXPERIMENT_TYPE}"
mkdir -p "${OUTPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# ==========================================
# Run Recovery Attack
#   - Comment this block to skip attack
# ==========================================
echo "[ATTACK] Start: $(date)"
python recover_og.py \
    --model_path "${MODEL_PATH}" \
    --benchmark_path "${BENCHMARK_PATH}" \
    --strategies ${STRATEGIES} \
    --output_dir "${OUTPUT_DIR}" \
    ${NUM_SAMPLES}
echo "[ATTACK] Done: $(date)"
echo ""

echo "=========================================="
echo "Recovery Attack Complete!"
echo "Results in: ${OUTPUT_DIR}"
echo "=========================================="
echo "End time: $(date)"
