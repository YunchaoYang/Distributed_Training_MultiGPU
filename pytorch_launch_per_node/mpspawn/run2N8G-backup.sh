#!/bin/bash

#SBATCH --partition=hpg-ai

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:8

#SBATCH --cpus-per-task=4

#SBATCH --time=08:00:00
#SBATCH --mem=24gb

#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

module load gcc/9.3.0 cuda/11.4.3 pytorch/1.10

#export NCCL_DEBUG=INFO  
# comment it if you are not debugging distributed parallel setup
#export NCCL_DEBUG_SUBSYS=ALL 
# comment it if you are not debugging distributed parallel setup

# find the ip-address of one of the node. Treat it as master
ip1=`hostname -I | awk '{print $2}'`
echo $ip1

# Store the master node’s IP address or hostname 
export MASTER_ADDR=$(hostname)
export MASTER_PORT=10213 # set PORT number

echo "r$SLURM_NODEID master: $MASTER_ADDR"

echo "r$SLURM_NODEID Launching python script"

srun python train.py --nodes=2 --ngpus 8 --ip_adress $ip1 --epochs 10
