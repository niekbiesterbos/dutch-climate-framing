#!/bin/bash
#SBATCH --job-name=exp18_classify
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/s4744497/thesis/logs/exp18_classify_%j.out
#SBATCH --error=/scratch/s4744497/thesis/logs/exp18_classify_%j.err

mkdir -p /scratch/s4744497/thesis/logs
source /scratch/s4744497/thesis/venv/bin/activate
python /scratch/s4744497/thesis/src/annotate/gold_motions_micro.py
