# === PATCHED SAMPLE.PY TO GENERATE MULTIPLE VIDEOS ===

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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main(args):
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    using_cfg = args.cfg_scale > 1.0 if hasattr(args, "cfg_scale") else False
    latent_size = args.image_size // 8
    args.latent_size = latent_size

    model = get_models(args).to(device)

    if args.use_compile:
        model = torch.compile(model)

    # Load model weights
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)

    model.eval()
    if args.use_fp16:
        model.to(dtype=torch.float16)
    model = model.to(device)

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="sd-vae-ft-mse").to(device)
    if args.use_fp16:
        vae = vae.to(dtype=torch.float16)

    if not os.path.exists(args.save_video_path):
        os.makedirs(args.save_video_path)

    for idx in range(args.num_samples):
        seed = torch.initial_seed() + idx
        generator = torch.Generator(device=device).manual_seed(seed)

        print(f"\n🚀 Sampling video {idx + 1}/{args.num_samples} with seed {seed}...")

        z = torch.randn(
            1, args.num_frames, 4, latent_size, latent_size,
            dtype=torch.float16 if args.use_fp16 else torch.float32,
            device=device, generator=generator
        )

        if using_cfg:
            z = torch.cat([z, z], 0)
            y = torch.randint(0, args.num_classes, (1,), device=device)
            y_null = torch.tensor([101] * 1, device=device)
            y = torch.cat([y, y_null], dim=0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, use_fp16=args.use_fp16)
            sample_fn = model.forward_with_cfg if hasattr(model, "forward_with_cfg") else model.forward
        else:
            sample_fn = model.forward
            model_kwargs = dict(y=None, use_fp16=args.use_fp16)

        if args.sample_method == 'ddim':
            samples = diffusion.ddim_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False,
                model_kwargs=model_kwargs, progress=True, device=device
            )
        elif args.sample_method == 'ddpm':
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False,
                model_kwargs=model_kwargs, progress=True, device=device
            )
        else:
            raise ValueError("Unsupported sampling method")

        if args.use_fp16:
            samples = samples.to(dtype=torch.float16)

        b, f, c, h, w = samples.shape
        samples = rearrange(samples, 'b f c h w -> (b f) c h w')
        samples = vae.decode(samples / 0.18215).sample
        samples = rearrange(samples, '(b f) c h w -> b f c h w', b=b)

        video_ = ((samples[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
        video_save_path = os.path.join(args.save_video_path, f'sample_{idx + 1}.mp4')
        print(f"📽 Saving video to: {video_save_path}")
        imageio.mimwrite(video_save_path, video_, fps=8, quality=9)

        # === Free memory ===
        del samples, z, video_, sample_fn
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/sky_sample.yaml")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--save_video_path", type=str, default="./sample_videos/")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--sample_method", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--num_samples", type=int, default=1, help="Number of videos to generate")
    args = parser.parse_args()

    omega_conf = OmegaConf.load(args.config)
    omega_conf.ckpt = args.ckpt
    omega_conf.save_video_path = args.save_video_path
    omega_conf.use_fp16 = args.use_fp16
    omega_conf.cfg_scale = args.cfg_scale
    omega_conf.sample_method = args.sample_method
    omega_conf.num_samples = args.num_samples

    main(omega_conf)
