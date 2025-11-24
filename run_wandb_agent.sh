#!/bin/bash
#BSUB -J wandb_sweep
#BSUB -o wandb_sweep-%J.out
#BSUB -e wandb_sweep-%J.err
#BSUB -W 24:00
#BSUB -R "rusage[mem=32000]"
#BSUB -n 4

# Request ANY GPU by allowing several GPU queues
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -q "gpua10 gpua40 gpua100 gpuv100"

module load python/3.10

cd ~/deep_learning_02456/Deep-Learning-02456-Project
source .venv/bin/activate

wandb agent s214995-danmarks-tekniske-universitet-dtu/Deep-Learning-02456-Project/i7hp58sr
