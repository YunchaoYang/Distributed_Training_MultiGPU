#!/bin/bash

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=8

#SBATCH --mem=100GB
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu

##SBATCH --gpus-per-task=2

#SBATCH --gpus=a100:8

#SBATCH --time=72:00:00

#SBATCH --output=job.%J.out
#SBATCH --error=job.%J.err

ml load singularity/3.7.4 cuda/11.4.3

#CONTAINER=/apps/nvidia/containers/pytorch/21.12-py3.sif
CONTAINER=/apps/nvidia/containers/modulus/modulus_v22.03.sif

export NCCL_DEBUG=INFO
# can be set to either OFF (default), INFO, or DETAIL
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1

#TRAINING_SCRIPT="$(realpath "$HOME/monai_multigpu/unet_training_ddp_slurm_torchlaunch.py")"
#TRAINING_CMD="$TRAINING_SCRIPT"
#echo "Running \"$TRAINING_CMD\" on each node..."

PT_LAUNCH_UTILS_PATH=`pwd`
source "${PT_LAUNCH_UTILS_PATH}/pt_multinode_helper_funcs.sh"

pwd; hostname; date
echo "Primary node: $PRIMARY"
echo "Primary TCP port: $PRIMARY_PORT"
echo "Secondary nodes: $SECONDARIES"

#PT_LAUNCH_SCRIPT=$(realpath "${PT_LAUNCH_UTILS_PATH}/run_on_node.sh")


srun --mpi=none singularity exec --nv --bind .:/mnt $CONTAINER python /mnt/ldc_2d.py
