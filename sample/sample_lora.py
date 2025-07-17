# === LoRA-AWARE SAMPLE SCRIPT PATCHED FOR LATTE ===
'''
import os
import sys
try:
    import utils
    from diffusion import create_diffusion
    from utils import find_model
except:
    sys.path.append(os.path.split(sys.path[0])[0])
    import utils
    from diffusion import create_diffusion
    from utils import find_model

import torch
import argparse
import torchvision

from einops import rearrange
from models import get_models
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
import imageio
from omegaconf import OmegaConf

from peft import LoraConfig, get_peft_model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main(args):
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    using_cfg = args.cfg_scale > 1.0
    latent_size = args.image_size // 8
    args.latent_size = latent_size

    model = get_models(args)

    # === Inject LoRA ===
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="custom",
        target_modules=["to_q", "to_k", "to_v", "proj"]
    )
    model = get_peft_model(model, lora_config)

    # === Load LoRA checkpoint only ===
    ckpt_path = args.ckpt
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "lora" in checkpoint:
        model.load_state_dict(checkpoint["lora"], strict=False)
    else:
        raise ValueError("❌ Provided checkpoint does not contain LoRA weights")

    model.eval()
    model = model.to(device)
    if args.use_fp16:
        print('WARNING: using half percision for inferencing!')
        model = model.half()

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(device)
    if args.use_fp16:
        vae = vae.to(dtype=torch.float16)

    # === Generate latent noise
    if args.use_fp16:
        z = torch.randn(1, args.num_frames, 4, latent_size, latent_size, dtype=torch.float16, device=device)
    else:
        z = torch.randn(1, args.num_frames, 4, latent_size, latent_size, device=device)

    # === Setup sampling config
    if using_cfg:
        z = torch.cat([z, z], 0)
        y = torch.randint(0, args.num_classes, (1,), device=device)
        y_null = torch.tensor([101] * 1, device=device)
        y = torch.cat([y, y_null], dim=0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, use_fp16=args.use_fp16)
        sample_fn = model.forward_with_cfg
    else:
        sample_fn = model.forward
        model_kwargs = dict(y=None, use_fp16=args.use_fp16)

    # === Perform sampling
    if args.sample_method == 'ddim':
        samples = diffusion.ddim_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
    elif args.sample_method == 'ddpm':
        samples = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )

    print(samples.shape)
    b, f, c, h, w = samples.shape
    samples = rearrange(samples, 'b f c h w -> (b f) c h w')
    samples = vae.decode(samples / 0.18215).sample
    samples = rearrange(samples, '(b f) c h w -> b f c h w', b=b)

    if not os.path.exists(args.save_video_path):
        os.makedirs(args.save_video_path)

    video_ = ((samples[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
    video_save_path = os.path.join(args.save_video_path, 'sample' + '.mp4')
    print(video_save_path)
    imageio.mimwrite(video_save_path, video_, fps=8, quality=9)
    print('save path {}'.format(args.save_video_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/sky_sample.yaml")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--save_video_path", type=str, default="./sample_videos/")
    parser.add_argument("--use_fp16", action="store_true")
    args = parser.parse_args()

    omega_conf = OmegaConf.load(args.config)
    omega_conf.ckpt = args.ckpt
    omega_conf.save_video_path = args.save_video_path
    omega_conf.use_fp16 = args.use_fp16
    main(omega_conf)

'''
# === LoRA-AWARE SAMPLE SCRIPT PATCHED FOR LATTE ===

import os
import sys
try:
    import utils
    from diffusion import create_diffusion
    from utils import find_model
except:
    sys.path.append(os.path.split(sys.path[0])[0])
    import utils
    from diffusion import create_diffusion
    from utils import find_model

import torch
import argparse
import torchvision

from einops import rearrange
from models import get_models
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
import imageio
from omegaconf import OmegaConf

from peft import LoraConfig, get_peft_model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main(args):
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    using_cfg = args.cfg_scale > 1.0
    latent_size = args.image_size // 8
    args.latent_size = latent_size

    model = get_models(args)

    # === Load pretrained base model (skytimelapse.pt) BEFORE LoRA injection ===
    pretrained_base_path = os.path.join("./checkpoints", "skytimelapse.pt")
    base_ckpt = torch.load(pretrained_base_path, map_location="cpu")
    if "ema" in base_ckpt:
        base_ckpt = base_ckpt["ema"]
    model.load_state_dict(base_ckpt, strict=False)

    # === Inject LoRA ===
    ckpt_path = args.ckpt
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "lora" in checkpoint:
        model = get_peft_model(model, LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="custom",
            target_modules=["to_q", "to_k", "to_v", "proj"]
        ))
        model.load_state_dict(checkpoint["lora"], strict=False)
    else:
        raise ValueError("❌ Provided checkpoint does not contain LoRA weights")

    model.eval()
    model = model.to(device)
    if args.use_fp16:
        print('WARNING: using half percision for inferencing!')
        model = model.half()

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(device)
    if args.use_fp16:
        vae = vae.to(dtype=torch.float16)

    # === Setup sampling config
    if using_cfg:
        sample_fn = model.forward_with_cfg
        y = torch.randint(0, args.num_classes, (1,), device=device)
        y_null = torch.tensor([101] * 1, device=device)
        y = torch.cat([y, y_null], dim=0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, use_fp16=args.use_fp16)
    else:
        sample_fn = model.forward
        model_kwargs = dict(y=None, use_fp16=args.use_fp16)

    # === Perform multiple video samplings
    if not os.path.exists(args.save_video_path):
        os.makedirs(args.save_video_path)

    for idx in range(args.num_samples):
        #seed = int(torch.initial_seed()) + idx
        #torch.manual_seed(seed)
        seed = int(torch.initial_seed()) + idx
        generator = torch.Generator(device=device).manual_seed(seed)
        

        print(f"Sampling video {idx + 1}/{args.num_samples} with seed {seed}...")
        print(f"\n🚀 Sampling video {idx + 1}/{args.num_samples}...")
        if args.use_fp16:
            #z = torch.randn(1, args.num_frames, 4, latent_size, latent_size, dtype=torch.float16, device=device)
            z = torch.randn(1, args.num_frames, 4, latent_size, latent_size, dtype=torch.float16 if args.use_fp16 else torch.float32, device=device, generator=generator)

        else:
            z = torch.randn(1, args.num_frames, 4, latent_size, latent_size, device=device)

        if using_cfg:
            z = torch.cat([z, z], 0)

        if args.sample_method == 'ddim':
            samples = diffusion.ddim_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
        elif args.sample_method == 'ddpm':
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
        else:
            raise ValueError("Unsupported sampling method")

        b, f, c, h, w = samples.shape
        samples = rearrange(samples, 'b f c h w -> (b f) c h w')
        samples = vae.decode((samples / 0.18215).to(dtype=torch.float16 if args.use_fp16 else torch.float32)).sample
        samples = rearrange(samples, '(b f) c h w -> b f c h w', b=b)

        video_ = ((samples[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
        video_save_path = os.path.join(args.save_video_path, f'sample_{idx + 1}.mp4')
        print(video_save_path)
        imageio.mimwrite(video_save_path, video_, fps=8, quality=9)
        print('save path {}'.format(args.save_video_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1, help="Number of videos to sample")
    parser.add_argument("--config", type=str, default="./configs/sky_sample.yaml")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--save_video_path", type=str, default="./sample_videos/")
    parser.add_argument("--use_fp16", action="store_true")
    args = parser.parse_args()

    omega_conf = OmegaConf.load(args.config)
    omega_conf.ckpt = args.ckpt
    omega_conf.save_video_path = args.save_video_path
    omega_conf.use_fp16 = args.use_fp16
    omega_conf.num_samples = args.num_samples
    main(omega_conf)
