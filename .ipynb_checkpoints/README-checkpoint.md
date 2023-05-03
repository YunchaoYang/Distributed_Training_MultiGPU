# MultiGPU_Training_Examples
### PyTorch 
  1. launch One process/per node in SLURM
    
     - use torchrun (torch.distributed.launch)
     - use mp.spawn
    
  2. launch directly One process/per GPU use srun
     - use srun

### PyTorch Lightning
  1. launch one process per GPU in srun

### MONAI
  1. use torch.distributed.launch
  2. launch per process use srun

### NVIDIA MODULUS
  launch per process use srun

### NVIDIA NeMo 