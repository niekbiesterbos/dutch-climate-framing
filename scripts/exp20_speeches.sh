#!/bin/bash
#SBATCH --job-name=exp20_speeches
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/s4744497/thesis/logs/exp20_speeches_%j.out
#SBATCH --error=/scratch/s4744497/thesis/logs/exp20_speeches_%j.err

mkdir -p /scratch/s4744497/thesis/logs
source /scratch/s4744497/thesis/venv/bin/activate
python /scratch/s4744497/thesis/src/score/macro_frame_speeches.py
