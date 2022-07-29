#!/bin/bash

# Scripts for  interactivel running PyTorch Distributed Data Parallel with multiple GPUs on a single Node.

# This is an independent script for launching multiple DDP process with torch.distributed.launch module on a single node on slurm-allocated resources. 
# Before running this script, make sure you are running it in the requested A100 multiGPU node either using OpenOnDemand (ood.rc.ufl.edu) or srun. 


###############################################################################################
###############################################################################################
#USER Definition

# to run 2 tasks on 1 node with 2 GPUs.
export GPUS_PER_TASK=2
export GPU_DEVICES_ID=0,1

CONTAINER=/red/ufhpc/hityangsir/MultiNode_MONAI_example/pyt21.07 #path/to/your/singularity/container
PYTHON_PATH="singularity exec --nv --bind $PWD:/mnt $CONTAINER python3"

TRAINING_SCRIPT="$(realpath "/mnt/unet_training_ddp_slurm_torchlaunch.py")" #path/to/training/code
TRAINING_CMD="$TRAINING_SCRIPT" # plus additional arguments/options 

#USER Definition end
###############################################################################################
###############################################################################################


pwd; hostname; date

# Global environment constants recommended by Nvidia
EXCLUDE_IB_LIST=mlx5_4,mlx5_5,mlx5_10,mlx5_11
export NCCL_DEBUG=WARN #INFO
export NCCL_IB_HCA=^${EXCLUDE_IB_LIST}
export NCCL_SOCKET_IFNAME=bridge-1145

module load singularity

export PRIMARY=localhost
export PRIMARY_PORT=12345  #######if conflict, change this port number between 1000-32767

echo "Primary node: $PRIMARY"
echo "Primary TCP port: $PRIMARY_PORT"

export CUDA_VISIBLE_DEVICES=$GPU_DEVICES_ID
echo "CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES

# For multiGPU on a single node 
LAUNCH_CMD="$PYTHON_PATH \
           -m torch.distributed.launch \
           --nproc_per_node=$GPUS_PER_TASK \
           --master_addr=$PRIMARY \
           --master_port=$PRIMARY_PORT \
            $TRAINING_CMD"

$LAUNCH_CMD
