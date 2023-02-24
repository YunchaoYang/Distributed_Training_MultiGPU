#!/bin/bash
#SBATCH --job-name=PLtest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2

##SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=64 
#SBATCH --mem=512gb
#SBATCH --time=50:00:00
#SBATCH --output=PLtest_%j.out
#SBATCH --partition=hpg-ai

#SBATCH --exclude=c1000a-s11,c1001a-s11


pwd; hostname; date

module load singularity

CONTAINER=/blue/ufhpc/hityangsir/test-nemo/debug2/nemo:22.11

export NCCL_NET_PLUGIN=none
export NCCL_DEBUG=INFO

export PYTHONFAULTHANDLER=1

EXCLUDE_IB_LIST=mlx5_4,mlx5_5,mlx5_10,mlx5_11
export NCCL_IB_HCA=^${EXCLUDE_IB_LIST}
export NCCL_SOCKET_IFNAME=bridge-1145

export HYDRA_FULL_ERROR=1 


export SINGULARITYENV_TMPDIR=$PWD/tmp
export SLURM_TMPDIR=$PWD/tmp
export TMPDIR=$PWD/tmp

export WORLD_SIZE=2 ###########################################################################

set | grep SLURM | while read line; do echo "# $line"; done

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

nvidia-smi

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES


srun singularity exec --nv $CONTAINER \
    python3 /blue/ufhpc/hityangsir/test-nemo/debug2/tp1/PL_multiGPU/backbone_image_classifier.py \
    --trainer.accelerator 'gpu' \
    --trainer.devices 2 \
    --trainer.strategy 'ddp'

