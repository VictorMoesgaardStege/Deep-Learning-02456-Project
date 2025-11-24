#!/bin/bash
#BSUB -J wandb_sweep                     # Job name
#BSUB -o wandb_sweep.%J.out              # STDOUT file
#BSUB -e wandb_sweep.%J.err              # STDERR file
#BSUB -W 24:00                           # Wall time (48 hours)
#BSUB -n 4                                # Number of CPU cores
#BSUB -R "rusage[mem=32GB]"               # Memory request
#BSUB -gpu "num=1"                        # Request 1 GPU
#BSUB -q gpuv100                          # Queue (adjust if needed)

# Load modules
module load python3

# Go to your project directory
cd ~/deep_learning_02456/Deep-Learning-02456-Project

# Activate your venv
source .venv/bin/activate

# Run the wandb agent for the sweep
wandb agent s214995-danmarks-tekniske-universitet-dtu/Deep-Learning-02456-Project/4dfjifzs
