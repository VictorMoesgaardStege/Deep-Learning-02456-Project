#!/bin/bash
#BSUB -J wandb_sweep                     # Job name
#BSUB -o wandb_sweep-%J.out              # Output log file
#BSUB -e wandb_sweep-%J.err              # Error log file
#BSUB -W 24:00                           # Wall time (24 hours)
#BSUB -R "rusage[mem=32000]"             # 32 GB RAM
#BSUB -n 4                               # 4 CPU cores

# Request 1 GPU and run job on a GPU queue
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -q gpuv100

module load python/3.10

cd ~/deep_learning_02456/Deep-Learning-02456-Project
source .venv/bin/activate

#wandb agent s214995-danmarks-tekniske-universitet-dtu/Deep-Learning-02456-Project/4dfjifzs
wandb agent s214995-danmarks-tekniske-universitet-dtu/Deep-Learning-02456-Project/i7hp58sr
