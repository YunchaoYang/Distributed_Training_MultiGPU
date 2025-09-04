# Multi-GPU Distributed Training

**Distributed training launch patterns with DDP, FSDP across PyTorch, PyTorch Lightning,  MONAI, NVIDIA Modulus, and NVIDIA NeMo, Deepspeed, Ray Train, tailored for SLURM and other multi-GPU environments.**

---

## 📂 Frameworks

| Framework / Method                     | Launch Strategy                        | Notes                             |
|---------------------------------------|----------------------------------------|-----------------------------------|
| **PyTorch (torchrun / torch.distributed)** | Launch one process per node,  via `torchrun`  or `torch.distributed.launch`   | DDP utility, cycleGAN, Monai      |
| **PyTorch (mp.spawn)**                     | Launch one process per node, process generation using `mp.spawn`  | Handy for script-based launches  |
| **PyTorch (srun)**                         | Launch one process per GPU via `srun`         | Clean integration with SLURM and srun   |
| **PyTorch (FSDP)**                         | One process per node via `torchrun`    | Standard FSDP      |
| **PyTorch Lightning**                      | One process per GPU via `srun`         | Uses Lightning “Trainer” class     |
| **MONAI (SLURM, DDP)**                     | `torch.distributed.launch` & `srun`    | Best practices for medical ML      |
| **NVIDIA Modulus**                         | Launch per process with `srun`         | PDE-oriented scientific workloads  |
| **NVIDIA NeMo Megatron**                   | Per process launching strategy         | Megatron model parallel for LLM training; singularity  |
| **Ray (SLURM Launch)**                     | Launch per process with `srun`         | general distributed communication framework  |
| **Deepspeed (SLURM Launch)**               | Launch per process with `srun`         | Deepspeed distributed training configuration  |

Each folder in this repo contains example scripts and job templates tailored to its framework.  

---
