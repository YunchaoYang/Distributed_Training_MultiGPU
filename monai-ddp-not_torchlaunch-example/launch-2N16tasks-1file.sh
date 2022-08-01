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
      
# If launch without torch.distributed.launch,
#       set #SBATCH --ntasks=total number of processes to run on all nodes
#       set #SBATCH --ntasks-per-node=--ntasks/--nodes    
#       set #SBATCH --gpus=--ntasks     
#       set #SBATCH --gpus-per-task=1
#       modify `LAUNCH_CMD` in `run_on_node.sh` to launch without torch.distributed.launch

# (c) 2021, Brian J. Stucky, UF Research Computing
# 2021/09, modified by Huiwen Ju, hju@nvidia.com

# Resource allocation.
#SBATCH --wait-all-nodes=1
##SBATCH --job-name=multinode_pytorch
##SBATCH --mail-type=ALL
##SBATCH --mail-user=<your email>

#SBATCH --nodes=2               # How many DGX nodes? Each has 8 A100 GPUs
#SBATCH --ntasks=16             # How many tasks? One per GPU
#SBATCH --ntasks-per-node=8     # Split 8 per node for the 8 GPUs
#SBATCH --gpus=16               # Total GPUs
##SBATCH --gpus-per-task=1       # 1 GPU per task
#SBATCH --cpus-per-task=4       # How many CPU cores per task, upto 16 for 8 tasks per node
#SBATCH --mem=96gb              # CPU memory per node--up to 1TB (Not GPU memory--that is 80GB per A100 GPU)
#SBATCH --partition=hpg-ai      # Specify the HPG AI partition

# Enable the following to limit the allocation to a single SU.
## SBATCH --constraint=su7
#SBATCH --time=48:00:00
#SBATCH --output=%x.%j.out

###########################################################################################
###########################################################################################
## USER definition

export NCCL_DEBUG=INFO

# Training command specification: training_script -args.
TRAINING_CMD="$(realpath "/mnt/unet_training_ddp_slurm.py")"

# Python location (if not provided, system default will be used).
# Here we run within a MONAI Singularity container based on NGC PyTorch container,
# see `build_container.sh` to build a MONAI Singularity container.

CONTAINER=/red/ufhpc/hityangsir/MultiNode_MONAI_example/pyt21.07
PYTHON_PATH="singularity exec --nv --bind $PWD:/mnt $CONTAINER python3"       

## USER definition end
###########################################################################################
###########################################################################################

##
EXCLUDE_IB_LIST=mlx5_4,mlx5_5,mlx5_10,mlx5_11
export NCCL_DEBUG=WARN
export NCCL_IB_HCA=^${EXCLUDE_IB_LIST}
export NCCL_SOCKET_IFNAME=bridge-1145

## 
get_unused_port() {
    # Well-known ports end at 1023.  On Linux, dynamic ports start at 32768
    # (see /proc/sys/net/ipv4/ip_local_port_range).
    local MIN_PORT=1024
    local MAX_PORT=32767

    local USED_PORTS=$(netstat -a -n -t | tail -n +3 | tr -s ' ' | \
        cut -d ' ' -f 4 | sed 's/.*:\([0-9]\+\)$/\1/' | sort -n | uniq)

    # Generate random port numbers within the search range (inclusive) until we
    # find one that isn't in use.
    local RAN_PORT
    while
        RAN_PORT=$(shuf -i 1024-32767 -n 1)
        [[ "$USED_PORTS" =~ $RAN_PORT ]]
    do
        continue
    done

    echo $RAN_PORT
}

init_node_info() {
    export PRIMARY=$(hostname -s)
    SECONDARIES=$(scontrol show hostnames $SLURM_JOB_NODELIST | \
        grep -v $PRIMARY)

    ALL_NODES="$PRIMARY $SECONDARIES"
    export PRIMARY_PORT=$(get_unused_port)
}

init_node_info # may not be necessary

pwd; hostname; date

echo "Primary node: $PRIMARY"
echo "Primary TCP port: $PRIMARY_PORT"
echo "Secondary nodes: $SECONDARIES"

PT_LAUNCH_SCRIPT="run_on_node.sh"
echo "Running \"$TRAINING_CMD\" on each node..."

module load singularity

LAUNCH_CMD="$PYTHON_PATH \
            $TRAINING_CMD"
srun  --export=ALL   --unbuffered $LAUNCH_CMD
