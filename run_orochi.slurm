#!/bin/bash
#SBATCH --job-name=uniud_large
#SBATCH --output=/home/mdurso/slurm_logs/build_%j.out
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=100GB

# 1. Load Conda
source "/home/mdurso/miniconda3/etc/profile.d/conda.sh"
conda activate vggt # colmap 3.12

# 2. Run Script
python run_for_dataset.py


echo "The end."