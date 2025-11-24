#!/bin/bash
#SBATCH --job-name=wandb_sweep
#SBATCH --output=wandb_sweep-%j.out
#SBATCH --time=24:00:00

# Request resources:
#SBATCH --gres=gpu:1            # GPU = huge speed boost
#SBATCH --cpus-per-task=4       # 4 CPU threads for data loading / numpy ops
#SBATCH --mem=32G               # Enough memory for large 3000-unit layers

# Load modules
module load python/3.10

# Activate your venv inside project folder
cd ~/deep_learning_02456/Deep-Learning-02456-Project
source .venv/bin/activate

# Run the wandb agent
wandb agent s214995-danmarks-tekniske-universitet-dtu/Deep-Learning-02456-Project/4dfjifzs
