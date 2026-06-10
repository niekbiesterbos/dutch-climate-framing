#!/bin/bash
#SBATCH --job-name=exp19_manifestos
#SBATCH --output=/scratch/s4744497/thesis/logs/exp19_manifestos_%j.out
#SBATCH --error=/scratch/s4744497/thesis/logs/exp19_manifestos_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

mkdir -p /scratch/s4744497/thesis/logs

source /scratch/s4744497/thesis/venv/bin/activate

python3 /scratch/s4744497/thesis/src/score/macro_frame_manifestos.py