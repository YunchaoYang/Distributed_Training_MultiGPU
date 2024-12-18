#!/bin/bash 
 
#SBATCH --job-name=code-server
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j_%N.log 

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                                                                                                                                                  
#SBATCH --cpus-per-task=128                                                                                                                                                  

#SBATCH --mem=32gb 
#SBATCH --partition=gpu
#SBATCH --gpus=geforce:1 # A100:1

#SBATCH --cpus-per-task=32


# 

PASSWORD=1234 # TODO: Change to secure password
PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "********************************************************************" 
echo "Starting code-server in Slurm"
echo "Environment information:" 
echo "Date:" $(date)
echo "Allocated node:" $(hostname)
echo "Node IP:" $(ip a | grep 131.159)
echo "Path:" $(pwd)
echo "Password to access VSCode:" $PASSWORD
echo "Listening on:" $PORT
echo "********************************************************************" 

PASSWORD=$PASSWORD code-server --bind-addr 0.0.0.0:$PORT --auth password --disable-telemetry
