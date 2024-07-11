
pip install -U ray[train] torch torchvision
Use conda env: torch-timm, python=3.8, ray=2.10

Use the ray.train.torch.prepare_model and ray.train.torch.prepare_data_loader utility functions to set up your model and data for distributed training. This automatically wraps the model with DistributedDataParallel and places it on the right device, and adds DistributedSampler to the DataLoaders.

Instantiate a TorchTrainer with 4 workers, and use it to run the new training function.

