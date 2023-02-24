# Multi-Nodes-Training with Pytorch Lightning using NeMo container

This is an intermediate test of debugging the p-tune and prompt tuning using NVIDIA NeMo container. 

In this test, we only test the Pytorch Lightning without using other required libraries by NeMo: including Hydra.

### Requirements:
- Pytorch Lightning 1.8.6 inside NeMo container 22.11 (nemo v1.14)

The python training code is from Pytorch Lightning GitHub repo with PL version 1.8.6.

"""MNIST backbone image classifier example."""

To run: python backbone_image_classifier.py --trainer.max_epochs=50

Reference: 

 - https://pytorch-lightning.readthedocs.io/en/stable/clouds/cluster_advanced.html 
