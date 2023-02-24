#!/bin/bash
#SBATCH --nodes 1             
#SBATCH --gres=gpu:2          # Request 2 GPU "generic resources”.
#SBATCH --tasks-per-node=2    # Request 1 process per GPU. You will get 1 CPU per process by default. Request more CPUs with the "cpus-per-task" parameter to enable multiple data-loader workers to load data in parallel.
#SBATCH --mem=8G      
#SBATCH --partition=hpg-ai

#SBATCH --time=0-03:00
#SBATCH --output=PLtest-%j.out

module load conda
conda activate /red/ufhpc/hityangsir/.conda/envs/pytorch_lightning

export NCCL_BLOCKING_WAIT=1 #Pytorch Lightning uses the NCCL backend for inter-GPU communication by default. Set this variable to avoid timeout errors.

srun python pytorch-ddp-test-pl.py  --batch_size 256
