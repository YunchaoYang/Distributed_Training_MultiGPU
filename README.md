# Multi-GPU Distributed Training

**Distributed training launch patterns across PyTorch, PyTorch Lightning, MONAI, NVIDIA Modulus, and NVIDIA NeMo, tailored for SLURM and other multi-GPU environments.**

---

## 📂 Frameworks

| Framework / Method                     | Launch Strategy                        | Notes                             |
|---------------------------------------|----------------------------------------|-----------------------------------|
| **PyTorch (torchrun / torch.distributed)** | One process per node via `torchrun`    | Standard DDP; straightforward      |
| **PyTorch (mp.spawn)**                     | One process per node using `mp.spawn`  | Handy for script-based launches    |
| **PyTorch + SLURM (`srun`)**               | One process per GPU via `srun`         | Clean integration with SLURM       |
| **PyTorch Lightning + SLURM**             | One process per GPU via `srun`         | Uses Lightning “Trainer” class     |
| **MONAI (SLURM launch)**                   | `torch.distributed.launch` & `srun`    | Best practices for medical ML      |
| **NVIDIA Modulus**                        | Launch per process with `srun`         | PDE-oriented scientific workloads  |
| **NVIDIA NeMo Megatron**                   | Per process launching strategy         | Optimized for LLM training stacks  |
| **Ray (SLURM Launch)**                     | Launch per process with `srun`         | general distributed communication framework  |
| **Deepspeed (SLURM Launch)**                     | Launch per process with `srun`         | general distributed communication framework  |

Each folder in this repo contains example scripts and job templates tailored to its framework.  
---

## 🔎 Overview

This repository hosts a variety of examples demonstrating how to launch and manage distributed training jobs using different frameworks and strategies.  

It is designed for:
- **Researchers** scaling workloads from a single node to multi-node GPU clusters.  
- **Engineers** experimenting with different frameworks for performance comparison.  
- **HPC users** integrating PyTorch, MONAI, NeMo, and Modulus into SLURM-based environments.  
