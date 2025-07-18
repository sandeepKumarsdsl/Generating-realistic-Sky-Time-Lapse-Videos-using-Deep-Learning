# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for Latte using PyTorch DDP.
"""

import torch
# Maybe use fp16 percision training need to set to False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import os
import math
import argparse

#import torch.distributed as dist
from glob import glob
from time import time
from copy import deepcopy
from einops import rearrange
from models import get_models
from datasets import get_dataset
#from models.clip import TextEmbedder
from diffusion import create_diffusion
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler
#from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils import (clip_grad_norm_, create_logger, update_ema, 
                   requires_grad, cleanup, create_tensorboard, 
                   write_tensorboard, setup_distributed, get_experiment_dir)

#from peft import get_peft_model, LoraConfig, TaskType, get_peft_model_state_dict

torch.cuda.empty_cache() # 1 - temporary fix for memory issue 
#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):

    # Setup DDP:
    #setup_distributed() # 11 - Fix for change order DDP before torch.cuda.is_available()

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    #device = torch.device("cuda")
    
    '''
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    seed = args.global_seed + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, local rank={local_rank}, seed={seed}, world_size={dist.get_world_size()}.")
    '''
    rank = 0
    local_rank = 0
    #rank = int(os.environ.get("RANK", 0))
    #local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", local_rank)

    seed = args.global_seed + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Only print world size after init_process_group
    #if dist.is_available() and dist.is_initialized(): # 12 - change for DDP
    #if dist.is_initialized():
    #    world_size = dist.get_world_size()
    world_size = 1 
    print(f"Starting rank={rank}, local rank={local_rank}, seed={seed}, world_size={world_size}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        #model_string_name = args.model.replace("/", "-")  # 
        model_string_name = args.model # for naming folders
        num_frame_string = 'F' + str(args.num_frames) + 'S' + str(args.frame_interval)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-{num_frame_string}-{args.dataset}"  # Create an experiment folder
        experiment_dir = get_experiment_dir(experiment_dir, args)
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        tb_writer = create_tensorboard(experiment_dir)
        OmegaConf.save(args, os.path.join(experiment_dir, 'config.yaml'))
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)
        tb_writer = None

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    sample_size = args.image_size // 8
    args.latent_size = sample_size
    model = get_models(args)
    #########################################
    #         Inject LoRA adapters          #
    #########################################
    '''
    lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="custom",
    #target_modules=["qkv", "proj"],
    target_modules=["to_q", "to_k", "to_v", "proj"]
    )
    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()


    #########################################
    #########################################
    #          Freeze base model            #
    #########################################
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False
    '''
    #########################################
    
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="sd-vae-ft-mse").to(device)

    # # use pretrained model?
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
        if "ema" in checkpoint:  # supports checkpoints from train.py
            logger.info('Using ema ckpt!')
            checkpoint = checkpoint["ema"]

        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {}
        for k, v in checkpoint.items():
            if k in model_dict:
                pretrained_dict[k] = v
            else:
                logger.info('Ignoring: {}'.format(k))
        logger.info('Successfully Load {}% original pretrained model weights '.format(len(pretrained_dict) / len(checkpoint.items()) * 100))
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info('Successfully load model at {}!'.format(args.pretrained))

    if args.use_compile:
        model = torch.compile(model)
      
    # Note that parameter initialization is done within the tvdm constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
  
    if args.enable_xformers_memory_efficient_attention:
        logger.info("Using Xformers!")
        model.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        logger.info("Using gradient checkpointing!")
        model.enable_gradient_checkpointing()

    if args.fixed_spatial:
        trainable_modules = (
        "attn_temp",
        )
        model.requires_grad_(False)
        for name, module in model.named_modules():
            if name.endswith(tuple(trainable_modules)):
                for params in module.parameters():
                    logger.info("WARNING: Only train {} parametes!".format(name))
                    params.requires_grad = True
        logger.info("WARNING: Only train {} parametes!".format(trainable_modules))

    #assert dist.is_initialized(), "Process group was not initialized. Please check setup_distributed() and launch method."

    # set distributed training
    # 3 - model = DDP(model.to(device), device_ids=[local_rank]) 
    #if dist.is_available() and dist.is_initialized():
    #    model = DDP(model.to(device), device_ids=[local_rank])
    #else:
    model = model.to(device)


    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)

    # Setup data:
    dataset = get_dataset(args)

    '''  # 4 - DDP Fix  
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    if dist.is_initialized():
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=True,
            seed=args.global_seed
        )
    else:
    '''
    sampler = None
    loader = DataLoader(
        dataset,
        batch_size=int(args.local_batch_size),
        #shuffle=False, #5 - Shuffle fix for None
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} videos ({args.data_path})")

    # Scheduler
    lr_scheduler = get_scheduler(
        name="constant",
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare models for training:
    #update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights # 6 - Model fix for DDP
    update_ema(ema, model.module if hasattr(model, "module") else model, decay=0)

    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    first_epoch = 0
    start_time = time()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(loader))
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    print(f"num_training_epochs:{num_train_epochs}; max_train_steps:{args.max_train_steps}")
    # Potentially load in the weights and states from a previous save
    '''if args.resume_from_checkpoint:
        # TODO, need to checkout
        # Get the most recent checkpoint
        dirs = os.listdir(os.path.join(experiment_dir, 'checkpoints'))
        dirs = [d for d in dirs if d.endswith("pt")]
        dirs = sorted(dirs, key=lambda x: int(x.split(".")[0]))
        path = dirs[-1]
        logger.info(f"Resuming from checkpoint {path}")
        model.load_state(os.path.join(dirs, path))
        train_steps = int(path.split(".")[0])

    '''
    resume_path = None
    if args.resume_from_checkpoint:
        if isinstance(args.resume_from_checkpoint, str) and args.resume_from_checkpoint.endswith('.pt'):
            resume_path = args.resume_from_checkpoint  # Use manually provided path
        else:
            ckpt_dir = os.path.join(experiment_dir, 'checkpoints')
            if os.path.exists(ckpt_dir):
                dirs = [d for d in os.listdir(ckpt_dir) if d.endswith("pt")]
                if dirs:
                    dirs = sorted(dirs, key=lambda x: int(x.split(".")[0]))
                    resume_path = os.path.join(ckpt_dir, dirs[-1])

    if resume_path:
        print(f"🔁 Resuming from checkpoint: {resume_path}")
        # Load your model/optimizer/etc.
        state = torch.load(resume_path, map_location="cpu")
        #model.load_state(state["model"])

        if isinstance(state, dict):
            if "model" in state:
                to_load = state["model"]
            elif "ema" in state:
                to_load = state["ema"]
            else:
                to_load = state  # raw weights

            if hasattr(model, "module"):
                model.module.load_state_dict(to_load)
            else:
                model.load_state_dict(to_load)

            print(f"✅ Loaded model from: {resume_path}")
        else:
            raise RuntimeError("Checkpoint format not recognized.")
        train_steps = int(resume_path.split("/")[-1].split(".")[0])
    else:
        print("🚀 No checkpoint found or resume disabled. Starting from scratch.")
        train_steps = 0


    first_epoch = train_steps // num_update_steps_per_epoch
    resume_step = train_steps % num_update_steps_per_epoch

    for epoch in range(first_epoch, num_train_epochs):
        #sampler.set_epoch(epoch) # 7 - Sampler None fix for DDP
        if sampler is not None:
            sampler.set_epoch(epoch)
        for step, video_data in enumerate(loader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                continue

            x = video_data['video'].to(device, non_blocking=True)
            video_name = video_data['video_name']
            
            if args.dataset == "ucf101_img":
                image_name = video_data['image_name']
                image_names = []
                for caption in image_name:
                    single_caption = [int(item) for item in caption.split('=====')]
                    image_names.append(torch.as_tensor(single_caption))
            
            # x = x.to(device)
            # y = y.to(device) # y is text prompt; no need put in gpu
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                b, _, _, _, _ = x.shape
                x = rearrange(x, 'b f c h w -> (b f) c h w').contiguous()
                #x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                
                latent = vae.encode(x).latent_dist.sample()
    
                # 🔍 Monitor latent stats (before scaling)
                latent_mean = latent.mean().item()
                latent_std = latent.std().item()
                if train_steps % 400 == 0 and rank == 0:
                    print(f"[step {train_steps}] Latent mean: {latent_mean:.4f}, std: {latent_std:.4f}")

                x = latent.mul_(0.18215)
                            
                            
                
                torch.cuda.empty_cache() # 2 - temporary fix for memory issue
                x = rearrange(x, '(b f) c h w -> b f c h w', b=b).contiguous()
            
            if args.extras == 78: # text-to-video
                raise 'T2V training are Not supported at this moment!'
            elif args.extras == 2:
                if args.dataset == "ucf101_img":
                    model_kwargs = dict(y=video_name, y_image=image_names, use_image_num=args.use_image_num) # tav unet
                else:
                    model_kwargs = dict(y=video_name) # tav unet
            else:
                model_kwargs = dict(y=None, use_image_num=args.use_image_num)
            
            #model_kwargs = dict(y=None, use_image_num=args.use_image_num)

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean() / args.gradient_accumulation_steps
            loss.backward()
            #Mixed Precision Training (fp16)
            #scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision) #1

            #with torch.cuda.amp.autocast(enabled=args.mixed_precision):#2
            #    loss = loss_dict["loss"].mean() / args.gradient_accumulation_steps#3

            #scaler.scale(loss).backward()

            if train_steps < args.start_clip_iter: # if train_steps >= start_clip_iter, will clip gradient
                #gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=False) # 8 - Model fix for DDP
                gradient_norm = clip_grad_norm_(model.parameters(), args.clip_max_norm, clip_grad=False)
            else:
                #gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=True) # 9 - Model fix for DDP
                gradient_norm = clip_grad_norm_(model.parameters(), args.clip_max_norm, clip_grad=True)

            lr_scheduler.step()
            if train_steps % args.gradient_accumulation_steps == 0 and train_steps > 0:
                opt.step()
                opt.zero_grad()
                #scaler.step(opt)#1 - mix -precision
                #scaler.update()#2
                #opt.zero_grad()#3
                #update_ema(ema, model.module) # 10 - Model fix for DDP
                update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                #dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() #/ dist.get_world_size()
                # logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                logger.info(f"(step={train_steps:07d}/epoch={epoch:04d}) Train Loss: {avg_loss:.4f}, Gradient Norm: {gradient_norm:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                write_tensorboard(tb_writer, 'Train Loss', avg_loss, train_steps)
                write_tensorboard(tb_writer, 'Gradient Norm', gradient_norm, train_steps)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()
            
            # Save tvdm checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        # "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        # "opt": opt.state_dict(),
                        # "args": args
                    }

                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                #dist.barrier()
            '''

            ###############################################
            #                Lora checkpoints             #
            ###############################################
            
            if train_steps % args.ckpt_every == 0:
                if rank == 0:
                    checkpoint = {
                        #"lora": get_peft_model_state_dict(model)
                        "lora": get_peft_model_state_dict(model.module if hasattr(model, "module") else model)

                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved LoRA checkpoint to {checkpoint_path}")
                dist.barrier()
                '''

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train tvdm-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/sky_config/sky_train.yaml")
    args = parser.parse_args()
    main(OmegaConf.load(args.config))
