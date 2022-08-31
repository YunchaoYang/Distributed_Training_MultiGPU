#!/bin/bash -l

# SLURM SUBMIT SCRIPT

#SBATCH --nodes=2
#SBATCH --partition=hpg-ai
#SBATCH --gres=gpu:a100:8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32gb
#SBATCH --time=0-02:00:00

# (optional) debugging flags
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# (optional) set the network interface name manually
# export NCCL_SOCKET_IFNAME=lo


# load your Python environment, for example with conda
module load conda

conda activate pytorch_lightning

#pip install "jsonargparse[all]" 
# How to make it work
# 1. pip install jsonargparse
# 2. pip install docstring-parser

pip list

# (optional) run script with fake data for debugging
srun python train.py fit --trainer.num_nodes 2 --trainer.gpus 8

# run full training with ImageNet data already downloaded
srun python train.py fit --trainer.num_nodes 2 --trainer.gpus 8 --data_path imagenet_home_dir
