#!/bin/bash

if [ -z "$HEAD_NODE_IP"]; then  
    HEAD_NODE_IP=$1 # if not set, use first input argument
fi

if [ -z "$HEAD_NODE_PORT"]; then  
    HEAD_NODE_PORT=23458 # if not set, set default
fi


bash -c 'echo "SLURM_PROCID=${SLURM_PROCID} on $(hostname) ${HEAD_NODE_IP}:${HEAD_NODE_PORT}"'

#optional in case for additional requirement.
#module load conda
#conda activate torch-timm

python main.py \
     -a resnet50 \
     --dist-url "tcp://"${HEAD_NODE_IP}:${HEAD_NODE_PORT} \
     --dist-backend nccl \
     --multiprocessing-distributed \
     --world-size ${SLURM_NNODES} \
     --rank ${SLURM_PROCID} \
     --epoch 10 \
     --batch-size 256 \
     imagenet \
     
