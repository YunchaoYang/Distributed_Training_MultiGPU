#!/bin/bash

# This Script is to use torchrun launch benchmark run; it is a distributed data parallel training with ResNet50 on UF HiPerGator's AI system on ImageNet1k dataset,
# this script uses a conda  environment runtime.
# 
# This script uses `submit_1node-8GPUs_SLURM.sh` to launch the SLRUM job. Each job step is contained in `run_on_node.sh`.
#
# When launch with torchrun, 
#       set #SBATCH --ntasks=--nodes
#       set #SBATCH --ntasks-per-node=1  
#       set #SBATCH --gpus=total number of processes to run on all nodes
#       set #SBATCH --gpus-per-task=--gpus / --ntasks  
#       modify `resources`, and 

# modify `run_on_node.sh` for input arguments to the python training script

# 07/2024, Yunchao Yang, UF Research Computing

# Resource allocation.

#SBATCH --nodes=1
#SBATCH --ntasks=1              
#SBATCH --ntasks-per-node=1   # One task per GPU

#SBATCH --gpus=2
#SBATCH --gpus-per-task=2       #GPU per srun step task
#SBATCH --partition=hpg-ai

#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb

#SBATCH --time=24:00:00

#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err


#######################################################################################
#######################################################################################
#export NCCL_DEBUG=INFO  # comment it if you are not debugging distributed parallel setup
#export NCCL_DEBUG_SUBSYS=ALL # comment it if you are not debugging distributed parallel setup


module load conda
conda activate torch-timm

# find the ip-address of the head node. Use it as master
HEAD_HOST_NAME=$(hostname | cut -d '.' -f 1)
HEAD_NODE_IP=$(hostname --ip-address)
echo "Head Node IP: $HEAD_NODE_IP"
echo "NODE: $SLURM_NODEID ${HEAD_HOST_NAME}, ${HEAD_NODE_IP}, Launching python script"

srun ./run-on-node.sh ${HEAD_NODE_IP}
