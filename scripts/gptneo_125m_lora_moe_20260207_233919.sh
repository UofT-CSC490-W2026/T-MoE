#!/bin/bash
#
# T-MoE Experiment: gptneo_125m_lora_moe
# Auto-generated: 20260207_233919
#
#SBATCH --job-name=gptneo_125m_lora_moe
#SBATCH --partition=csc420
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=outputs/logs/gptneo_125m_lora_moe/gptneo_125m_lora_moe_%j.out
#SBATCH --error=outputs/logs/gptneo_125m_lora_moe/gptneo_125m_lora_moe_%j.err

echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "======================================"

# Environment variables
export PYTHONUNBUFFERED=1
export WANDB_MODE=online

# Run training
python train.py --config gptneo_125m_lora

echo "======================================"
echo "Job finished at: $(date)"
echo "======================================"
