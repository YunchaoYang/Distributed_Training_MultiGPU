import torch
import time
from multiprocessing import Pool

# Your workload
def gpu_task(gpu_id, data):
    # Assign the GPU
    torch.cuda.set_device(gpu_id)
    
    # Simulate GPU workload (replace with actual computation)
    x = torch.tensor(data).cuda()
    y = x * x

    return (gpu_id, data, y.item())

if __name__ == '__main__':

    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs.")

    # Create job list (gpu_id, data) pairs
    jobs = [(i % num_gpus, i) for i in range(8)]  # 8 jobs across available GPUs

    with Pool(processes=num_gpus) as pool:
        results = pool.starmap(gpu_task, jobs)

    for gpu_id, input_data, result in results:
        print(f"GPU: {gpu_id},  input: {input_data} => output: {result}")

