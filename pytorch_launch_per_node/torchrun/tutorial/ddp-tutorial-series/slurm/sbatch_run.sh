#!/bin/bash

#SBATCH --job-name=multinode-example
#SBATCH --partition=hpg-ai
#SBATCH --nodes=4
##SBATCH --ntasks=4

#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --mem=4GB

##SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4

#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

module load conda
conda activate pytorch_lightning

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

srun torchrun \
--nnodes 4 \
--nproc_per_node 1 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \
../multinode_torchrun.py 50 10
