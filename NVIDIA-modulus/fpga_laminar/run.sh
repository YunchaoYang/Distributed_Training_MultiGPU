#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=8

#SBATCH --mem=1000GB
#SBATCH --cpus-per-task=4
#SBATCH --partition=hpg-ai

#SBATCH --gpus=a100:8

#SBATCH --time=72:00:00
#SBATCH --output=job.%J.out
#SBATCH --error=job.%J.err

ml load singularity/3.7.4 cuda/11.4.3

#CONTAINER=/apps/nvidia/containers/pytorch/21.12-py3.sif
CONTAINER=/apps/nvidia/containers/modulus/modulus_v22.03.sif

srun --unbuffered --mpi=none -n8 --ntasks-per-node 8 singularity exec --nv --bind .:/mnt $CONTAINER python /mnt/fpga_flow.py

# mpirun -np 2 singularity exec --nv --bind .:/mnt $CONTAINER python /mnt/fpga_flow.py # no mpirun command

