import torch
from dataset import HorseZebraDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator

## DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import builtins
import os
import socket

def train_one_epoch(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, args):
    
    H_reals = 0 # number of real horses
    H_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (zebra, horse) in enumerate(loop):

        #zebra = zebra.to(args.gpu) # DDP
        #horse = horse.to(args.gpu) # DDP
        zebra = zebra.cuda(non_blocking=True)
        horse = horse.cuda(non_blocking=True)
        
        # 1 Train Discriminators H and Z
        with torch.cuda.amp.autocast():  # float16
            fake_horse = gen_H(zebra)    # H_fake = F(Z)
            D_H_real = disc_H(horse)     # Z_fake = G(H)
            D_H_fake = disc_H(fake_horse.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real)) # real horse mse loss
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake)) # fake horse mse loss
            D_H_loss = D_H_real_loss + D_H_fake_loss # Discrimator loss for real horse and fake horse (gen from zebra)

            fake_zebra = gen_Z(horse)
            D_Z_real = disc_Z(zebra)
            D_Z_fake = disc_Z(fake_zebra.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss # Discrimator loss for real zebra and fake zebra(gen from horse) 

            # put it togethor
            D_loss = (D_H_loss + D_Z_loss)/2 # total Discrimnator loss

        # Backward, optimize and scheduler step
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc) # gradient update All reduce
        d_scaler.update()

        # 2 Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_horse)
            D_Z_fake = disc_Z(fake_zebra)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake)) # H_fake = F(Z)
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake)) # Z_fake = G(H)

            # cycle loss
            cycle_zebra = gen_Z(fake_horse) # Cyc_Z = G(F(Z))
            cycle_horse = gen_H(fake_zebra) # Cyc_H = F(G(H))
            cycle_zebra_loss = l1(zebra, cycle_zebra)
            cycle_horse_loss = l1(horse, cycle_horse)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_zebra = gen_Z(zebra)
            identity_horse = gen_H(horse)
            identity_zebra_loss = l1(zebra, identity_zebra)
            identity_horse_loss = l1(horse, identity_horse)

            # add all togethor
            G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_zebra_loss * config.LAMBDA_CYCLE
                + cycle_horse_loss * config.LAMBDA_CYCLE
                + identity_horse_loss * config.LAMBDA_IDENTITY
                + identity_zebra_loss * config.LAMBDA_IDENTITY
            )

        # Backward, optimize and scheduler step            
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if dist.get_rank() == 0:
            if idx % 100 == 0:
                save_image(fake_horse*0.5+0.5, f"saved_images/fakehorse_{idx}.png")
                save_image(fake_zebra*0.5+0.5, f"saved_images/fakezebra_{idx}.png")

        loop.set_postfix(H_real=H_reals/(idx+1), H_fake=H_fakes/(idx+1))



def parse_args():
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--gpu', default=-1, type=int, 
                        help='local rank for distributed training')

    args = parser.parse_args()

    return args

def main(args):
    
    
    ######################################################################################    
    # DDP setting
    ######################################################################################        
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1
    
    if args.distributed:
        if args.local_rank != -1: # for torch.distributed.launch in a single node.
            args.rank = args.local_rank # args.local_rank is assigned by torch.distributed.launch module
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
    ######################################################################################    
    ### DDP setting end ###
    ######################################################################################   
       
    disc_H = Discriminator(in_channels=3).to(config.DEVICE) # if horse is real 
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE) # if zebra is real
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE) # generate a zebra from a horse
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE) # generate a horse from a zebra

    
    
    ######################################################################################    
    # DDP setting
    ######################################################################################        
    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu) #model.cuda(args.gpu)

            disc_H.to(f'cuda:{args.gpu}')
            disc_H = DDP(disc_H, device_ids=[args.gpu])
            #model_without_ddp = model.module            
            disc_Z.to(f'cuda:{args.gpu}')
            disc_Z = DDP(disc_Z, device_ids=[args.gpu])            

            gen_Z.to(f'cuda:{args.gpu}')
            gen_Z = DDP(gen_Z, device_ids=[args.gpu])
            
            gen_H.to(f'cuda:{args.gpu}')
            gen_H = DDP(gen_H, device_ids=[args.gpu])

        else: 
            print("args.gpu is not set") 
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")
     # suppress printing if not on master gpu
    if args.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    ###################################################################################### 
    ### DDP setting end ###
    ######################################################################################
        
        
        
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr = config.LEARNING_RATE * args.world_size, #DDP
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr = config.LEARNING_RATE * args.world_size, #DDP
        betas=(0.5, 0.999),
    )

    
    # 
    L1 = nn.L1Loss() # cycle consistence loss
    mse = nn.MSELoss() # GAN loss

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H, disc_H, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Z, disc_Z, opt_disc, config.LEARNING_RATE,
        )

    dataset = HorseZebraDataset(
        root_horse=config.TRAIN_DIR+"/trainA", root_zebra=config.TRAIN_DIR+"/trainB", transform=config.transforms
    )
    val_dataset = HorseZebraDataset(
       root_horse= config.VAL_DIR+"/testA", root_zebra=config.VAL_DIR+"/testB", transform=config.transforms
    )
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                        dataset=dataset, 
                        #even_divisible=True, 
                        shuffle=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
                        dataset=val_dataset, 
                        #even_divisible=True, 
                        shuffle=True
        )
        
    train_loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=(train_sampler is None),
        num_workers=config.NUM_WORKERS,
        sampler=train_sampler,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=(val_sampler is None),
        sampler=val_sampler,
        pin_memory=True,
    )
    
    g_scaler = torch.cuda.amp.GradScaler() # gradient scaler
    d_scaler = torch.cuda.amp.GradScaler()
    
    if dist.get_rank() == 0:
        print("gen_H parameters:", sum(p.numel() for p in gen_H.parameters()))
        print("gen_Z parameters:", sum(p.numel() for p in gen_Z.parameters()))
        print("dis_H parameters:", sum(p.numel() for p in disc_H.parameters()))
        print("dis_H parameters:", sum(p.numel() for p in disc_Z.parameters()))

    for epoch in range(config.NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        train_one_epoch(disc_H, disc_Z, gen_Z, gen_H, train_loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, args)
        print("Epoch: ", epoch)
        # save to 4 different set of network weights
        if config.SAVE_MODEL:
            if dist.get_rank() == 0:
                save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
                save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
                save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
                save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)

if __name__ == "__main__":
    args = parse_args()
    main(args)
    print("Job finished")
