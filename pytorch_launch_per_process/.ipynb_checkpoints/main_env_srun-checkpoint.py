
## adapated form https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904
# Starting process using srun

import os
import builtins
import argparse
import torch
import numpy as np 
import random
import torch.distributed as dist
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from datetime import datetime
import socket

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='resnet18', type=str)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size per GPU')
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, 
                        help='start epoch number (useful on restarts)')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--workers', default=2, type=int, help='number of workers for data loading')
    parser.add_argument('--log_interval', default=100, type=int, help='log intervals')

    # DDP configs:
    parser.add_argument('--world-size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, 
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int, 
                        help='local rank for distributed training')
    args = parser.parse_args()
    return args
                                         
def main(args):

    print(os.environ["WORLD_SIZE"])
    # DDP setting
    if "WORLD_SIZE" in os.environ:
        print(os.environ["WORLD_SIZE"])
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1
    print(args.distributed, args.local_rank)
    
    if args.distributed:
        if args.local_rank != -1: # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank

        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler, but not using torch.distributed.launch
            print("use SLURM environment variables to DDP")
            args.rank = int(os.environ['SLURM_PROCID'])            
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
            args.gpu = int(os.environ["SLURM_LOCALID"]) # args.rank % torch.cuda.device_count() # both two ways work

    # parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK",  "LOCAL_RANK", "WORLD_SIZE")
    }

    print(f"PID=[{os.getpid()}], on Node:{socket.gethostname()} GPU:{args.gpu} initializing process group with: {env_dict}")

    dist.init_process_group(backend="nccl", init_method="env://")
    
    # suppress printing if not on master gpu
    if args.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

       
    ### model ###
    model = CNN()
    
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.

        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            #model.cuda(args.gpu)
            model.to(f'cuda:{args.gpu}')
            
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module

        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
            model_without_ddp = model.module
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")
        
    ### optimizer ###
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
  
    ### resume training if necessary ###
    #if args.resume:
    #    pass
    
    ### data ###
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    
    val_dataset = torchvision.datasets.MNIST(root='./data', 
                                             train=False, 
                                             download=True, 
                                             transform=transforms.ToTensor()) 

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                        train_dataset,
                        num_replicas=args.world_size,
                        rank=args.rank
        )

        val_sampler = torch.utils.data.distributed.DistributedSampler(
                      val_dataset,
                      num_replicas=args.world_size,
                      rank=args.rank
        )


    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=True)
    
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    
    torch.backends.cudnn.benchmark = True
    
    start = datetime.now()
    
    ### main loop ###
    for epoch in range(args.start_epoch, args.epochs):

        #random.seed(epoch)
        # fix sampling seed such that each gpu gets different part of dataset
        if args.distributed: 
            train_loader.sampler.set_epoch(epoch)
        
        # adjust lr if needed #        
        train_one_epoch(train_loader, model, criterion, optimizer, epoch, args)

        #if args.rank == 0: # only val and save on master node
        #    validate(val_loader, model, criterion, epoch, args)
            # save checkpoint if needed #
    
    if args.rank == 0:
        print("Training complete in: " + str(datetime.now() - start))
     
def train_one_epoch(train_loader, model, criterion, optimizer, epoch, args):
    # pass
    # only one gpu is visible here, so you can send cpu data to gpu by 
    # input_data = input_data.cuda() as normal

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #data, target = data.to(args.gpu), target.to(args.gpu)
        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output, _ = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0 and args.gpu == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data) * args.world_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

            
def validate(val_loader, model, criterion, epoch, args):
    pass

if __name__ == '__main__':
    args = parse_args()
    main(args)

    print("Job finished")
