#!/bin/bash
#
# Script to launch a multi-gpu raytune HPO
# on UF HiperGator's AI partition
#
# Reference: https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html
# Modify from the NERSC repo https://github.com/NERSC/slurm-ray-cluster
#  
# (c) 2023 Huiwen Ju NVIDIA for monai integration      
# (c) 2024 Yunchao Yang, UF Research Computing

#SBATCH --job-name=test_RayTune
#SBATCH --time=02:00:00
#SBATCH --output=%x.%j.out
#SBATCH --partition=hpg-ai

### This script works for any number of nodes, Ray will find and manage all resources
### Give all resources on each node to a single Ray task, ray can manage the resources internally
# Requested 2 nodes with 2 GPU/node = total 4GPUs

#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --mem=200gb
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=4


module load conda
conda activate torch-timm

##############################################################################################
# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)


# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

##############################################################################################
port=6379

ip_head=$head_node_ip:$port
export ip_head
echo "*****************"
echo "IP Head: $ip_head"
echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block &
# By backgrounding the above srun task, we can proceed to start the Ray worker runtimes.

sleep 10 # Optional may be useful in certain versions of Ray < 1.0.

##############################################################################################
# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))
echo worker_num=${worker_num}
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" \
	--num-gpus "${SLURM_GPUS_PER_TASK}"
        --block &
    sleep 5
done

##############################################################################################

#### call your code below
python examples/mnist_pytorch_trainable.py --cuda

exit
