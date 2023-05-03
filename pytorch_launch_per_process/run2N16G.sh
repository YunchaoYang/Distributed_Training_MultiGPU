#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --partition=hpg-ai
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus=16 

#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb

#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

module load cuda/11.4.3 pytorch/1.10

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
#export WORLD_SIZE=$((SLURM_JOB_NUM_NODES*SLURM_GPUS_PER_NODE))
echo SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES
echo SLURM_TASKS_PER_NODE=$SLURM_TASKS_PER_NODE # 8(x2)

# export WORLD_SIZE=$((SLURM_JOB_NUM_NODES*SLURM_TASKS_PER_NODE)) #error
export WORLD_SIZE=$SLURM_NTASKS

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2

echo "NODELIST="${SLURM_NODELIST}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

echo "MASTER_ADDR="$MASTER_ADDR

echo "$SLURM_NODEID Launching job script"
echo "WORLD SIZE = $WORLD_SIZE"

EXCLUDE_IB_LIST=mlx5_4,mlx5_5,mlx5_10,mlx5_11
export NCCL_DEBUG=WARN #INFO #WARN
export NCCL_IB_HCA=^${EXCLUDE_IB_LIST}
export NCCL_SOCKET_IFNAME=bridge-1145

### init virtual environment if needed
#source ~/anaconda3/etc/profile.d/conda.sh
#conda activate myenv

### the command to run
srun --export=ALL python main_env_srun.py --epochs 20 

