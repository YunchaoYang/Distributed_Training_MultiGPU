#PyTorch Profiler

##refs
1. [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
2. [Understanding GPU Memory 1: Visualizing All Allocations over Time](https://pytorch.org/blog/understanding-gpu-memory-1/)
3. [Using PyTorch Profiler with DeepSpeed for performance debugging](https://www.deepspeed.ai/tutorials/pytorch-profiler/)

## How to run Profiler

## How to Run Memory Debuuging

`ml pytorch/2.2`

1. `python ResNet50MemorySnapshot.py`
output $HOSTName_dates.pickle

2. `python ResNet50MemoryProfiler.py`
output .json.gz .html
