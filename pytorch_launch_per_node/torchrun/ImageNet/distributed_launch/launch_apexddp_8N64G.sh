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

#SBATCH --nodes=8               # How many DGX nodes? Each has 8 A100 GPUs
#SBATCH --ntasks=8              # How many tasks? One per GPU
#SBATCH --ntasks-per-node=1     # Split 8 per node for the 8 GPUs
#SBATCH --gpus-per-task=8       # #GPU per srun step task
##SBATCH --gpus=8                # Total GPUs

#SBATCH --cpus-per-task=12       # How many CPU cores per task, upto 16 for 8 tasks per node
#SBATCH --mem=1024gb             # CPU memory per node--up to 1TB (Not GPU memory--that is 80GB per A100 GPU)
#SBATCH --partition=hpg-ai      # Specify the HPG AI partition

# Enable the following to limit the allocation to a single SU.
## SBATCH --constraint=su7
#SBATCH --time=48:00:00
#SBATCH --output=%x.%j.out

#SBATCH --account=bala-gatorflow
#SBATCH --qos=bala-gatorflow
#SBATCH --reservation=gatorflow

############################################################################
############################################################################
## USER customize##

# Singularity container and Python interpreter
# Here we run within a MONAI Singularity container based on NGC PyTorch container,
# see `build_container.sh` to build a MONAI Singularity container.
#CONTAINER=/red/ufhpc/hityangsir/MultiNode_MONAI_example/pyt21.07 # path/to/your/container
#CONTAINER=/apps/nvidia/containers/monai/core/0.9.0
#PYTHON_PATH="singularity exec --nv --bind $PWD:/mnt $CONTAINER python3"       

module load ngc-pytorch/1.11.0
PYTHON_PATH=python3

# Training command specification: training_script -args.
TRAINING_SCRIPT=main_amp.py
TRAINING_CMD="$TRAINING_SCRIPT -a resnet50 --b 224 --workers 4 --opt-level O2 ./" 

# Location of the PyTorch launch utilities, i.e. `pt_multinode_helper_funcs.sh` & `run_on_node.sh`.
PT_LAUNCH_UTILS_PATH=$PWD/utils

#USER definition end#
############################################################################
############################################################################

export NCCL_DEBUG=WARN #change to INFO if debugging DDP

source "${PT_LAUNCH_UTILS_PATH}/pt_multinode_helper_funcs.sh"
init_node_info

echo "Primary node: $PRIMARY"
echo "Primary TCP port: $PRIMARY_PORT"
echo "Secondary nodes: $SECONDARIES"

PT_LAUNCH_SCRIPT="./utils/run_on_node.sh"
echo "Running \"$TRAINING_CMD\" on each node..."

pwd; hostname; date
srun --unbuffered --export=ALL "$PT_LAUNCH_SCRIPT" "${PT_LAUNCH_UTILS_PATH}" "$TRAINING_CMD" "$PYTHON_PATH"
