#!/bin/bash
#SBATCH --wait-all-nodes=1
#SBATCH --nodes=2               
#SBATCH --ntasks=2              
#SBATCH --ntasks-per-node=1     
#SBATCH --gpus=16
#SBATCH --gpus-per-task=8     
#SBATCH --cpus-per-task=16       
#SBATCH --mem=64gb             
#SBATCH --partition=hpg-ai
#SBATCH --exclusive
#SBATCH --time=1:00:00
#SBATCH --output=%x.%j.out
#SBATCH --reservation=el8-hpgai
pwd; hostname; date

#############################################################################
# 1. Prep env
export USE_CONTAINER=true

if $USE_CONTAINER; then
    CONTAINER_NAME="/blue/vendor-nvidia/hju/monaicore1.2.0"
    BIND_PATH="/blue/vendor-nvidia/hju/data/hovernet_data/CoNSeP:/mnt"
    PREP_PATH="singularity exec --nv --bind $BIND_PATH $CONTAINER_NAME"
    echo "Use container"
else
    module load conda
    export ENV_NAME="torch-timm"
    conda activate $ENV_NAME
    PREP_PATH=""
fi
echo "PREP_PATH=$PREP_PATH"


#############################################################################
# 2. training command 
# Training command specification: training_script -args.
TRAINING_SCRIPT="$(realpath "training.py")"
TRAINING_CMD="$TRAINING_SCRIPT \
--stage=0 \
--ep=3 \
--bs=16 \
--log-dir=./logs \
--root=/mnt/Prepared"
echo "Running \"$TRAINING_CMD\" on each node..."


#############################################################################
# 3. torchrun args
# DO Not touch 
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
#head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
head_node_ip=$(hostname --ip-address)
echo Node IP: $head_node_ip

# Returns a random, unused TCP port number
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

export PRIMARY_PORT=$(get_unused_port)

TORCHRUN_ARGS="--nnodes=$SLURM_JOB_NUM_NODES \
     --nproc_per_node=$SLURM_GPUS_PER_TASK \
     --rdzv_id $RANDOM \
     --rdzv_backend c10d \
     --rdzv_endpoint $head_node_ip:$PRIMARY_PORT"

echo "torchrun args:$TORCHRUN_ARGS"

#############################################################################
# 4 other debug or info env variables 
export NCCL_DEBUG=INFO
# can be set to either OFF (default), INFO, or DETAIL
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1




#############################################################################
#############################################################################
# The structure is
# 1 Container_Env_Prep
# 2 torchrun + torchrun_args
# 3 training_file.py + train_args 
SUBMIT_SCRIPT="$PREP_PATH torchrun $TORCHRUN_ARGS  $TRAINING_CMD"

echo $SUBMIT_SCRIPT

srun --export=ALL $SUBMIT_SCRIPT

