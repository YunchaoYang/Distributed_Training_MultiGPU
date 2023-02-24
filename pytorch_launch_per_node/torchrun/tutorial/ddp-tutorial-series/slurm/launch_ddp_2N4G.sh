#!/bin/bash

# Script to launch a multi-node pytorch.distributed training run on UF HiPerGator's AI partition,
# a SLURM cluster using Singularity as container runtime.
# 
# This script uses `pt_multinode_helper_funcs.sh` and `run_on_node.sh`.
#
# If launch with torch.distributed.launch, 
#       set #SBATCH --ntasks=--nodes
#       set #SBATCH --ntasks-per-node=1  
#       set #SBATCH --gpus=total number of processes to run on all nodes
#       set #SBATCH --gpus-per-task=--gpus / --ntasks  
#       modify `LAUNCH_CMD` in `run_on_node.sh` to launch with torch.distributed.launch

# (c) 2021, Brian J. Stucky, UF Research Computing
# 2021/09, modified by Huiwen Ju, hju@nvidia.com
# 07/2022, modified by Yunchao Yang, UF Research Computing

# Resource allocation.
#SBATCH --wait-all-nodes=1

#SBATCH --nodes=2               # How many DGX nodes? Each has 8 A100 GPUs
#SBATCH --ntasks=2              # How many tasks? One per GPU
#SBATCH --ntasks-per-node=1     # Split 8 per node for the 8 GPUs
#SBATCH --gpus-per-task=2       # #GPU per srun step task

##SBATCH --gpus=4                # Total GPUs

#SBATCH --cpus-per-task=4       # How many CPU cores per task, upto 16 for 8 tasks per node
#SBATCH --mem=24gb             # CPU memory per node--up to 1TB (Not GPU memory--that is 80GB per A100 GPU)
#SBATCH --partition=hpg-ai      # Specify the HPG AI partition

# Enable the following to limit the allocation to a single SU.
#SBATCH --time=48:00:00
#SBATCH --output=%x.%j.out

# export NCCL_DEBUG=WARN #change to INFO if debugging DDP

module load pytorch/1.10
# module load conda
# conda activate pytorch_lightning

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}

head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip

#head_node_ip=`hostname -I | awk '{print $2}'`
head_node_ip=`hostname --ip-address`

echo `hostname -I` 
echo Node IP: $head_node_ip

export LOGLEVEL=INFO

pwd; hostname; date

srun --export=ALL torchrun \
--nnodes 2 \
--nproc_per_node 2 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \
multigpu_torchrun.py 50 10

