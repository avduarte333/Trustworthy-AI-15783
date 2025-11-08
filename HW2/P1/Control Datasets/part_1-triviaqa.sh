#!/bin/bash
#SBATCH --job-name=triviaqa_part_1
#SBATCH --partition=general
#SBATCH --gres=gpu:6000Ada:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out


# Load environment
source ~/miniconda3/bin/activate cmu

# Move to your project directory
cd /home/aduarte/Trustworthy_AI/HW2

# Run your code
python /home/aduarte/Trustworthy_AI/HW2/part_1-triviaqa.py