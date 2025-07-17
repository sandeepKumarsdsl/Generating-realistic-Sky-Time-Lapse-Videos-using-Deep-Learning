'''
import os
import argparse
import random
import torchvision
from tqdm import tqdm
import torch
from torchvision import transforms

from datasets.video_transforms import ToTensorVideo, CenterCropResizeVideo
from datasets.sky_timelapse.video_folder import VideoFolder

def save_videos_from_dataset(args):
    """
    Convert real frame folders into .mp4 clips with a fixed frame_interval,
    frames_per_video, and resolution.  Saves exactly `num_videos` clips.

    Expected input layout:
        args.data_root/<video_name>/<frame_folder>/*.jpg
    """
    # -------- transforms identical to training --------
    transform = transforms.Compose([
        CenterCropResizeVideo(args.resolution),  # custom in datasets.video_transforms
        ToTensorVideo(),                         # (T, C, H, W) in [0,1]
        transforms.Normalize([0.5]*3, [0.5]*3)   # -> [-1,1]
    ])

    # -------- dataset loader --------
    dataset = VideoFolder(
        root=args.data_root,
        nframes=args.frames_per_video * args.frame_interval,
        transform=None                # we apply transform manually after permute
    )
    print(f"✅ Loaded {len(dataset)} total candidate clips from {args.data_root}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Random subset
    indices = random.sample(range(len(dataset)),
                            min(args.num_videos, len(dataset)))

    saved = 0
    for clip_idx in indices:
        clip_tensor, _ = dataset[clip_idx]       # (C, T, H, W)

        # Re‑order to (T, C, H, W) for video_transforms
        clip_tensor = clip_tensor.permute(1, 0, 2, 3)

        # Sample every Nth frame
        clip_tensor = clip_tensor[::args.frame_interval]

        # Ensure we still have enough frames
        if clip_tensor.shape[0] < args.frames_per_video:
            continue
        clip_tensor = clip_tensor[:args.frames_per_video]  # keep first T

        # Apply transforms (crop, resize, normalize)
        clip_tensor = transform(clip_tensor)               # still (T,C,H,W)

        # Denormalize to uint8 for saving
        video_uint8 = ((clip_tensor * 0.5 + 0.5)            # [-1,1] -> [0,1]
                       .clamp(0, 1) * 255)                  # [0,255]
        video_uint8 = video_uint8.byte().permute(0, 2, 3, 1).cpu()  # T,H,W,C

        out_path = os.path.join(args.output_dir,
                                f"real_{saved:04d}.mp4")
        torchvision.io.write_video(out_path, video_uint8,
                                   fps=args.fps)
        saved += 1
        if saved % 50 == 0 or saved == args.num_videos:
            print(f"[{saved}/{args.num_videos}] saved {out_path}")
        if saved >= args.num_videos:
            break

    print(f"🎬 Finished: {saved} videos written to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Path to the skytimelapse dataset root')
    parser.add_argument('--output_dir', type=str, default='./real_videos', help='Directory to save output videos')
    parser.add_argument('--num_videos', type=int, default=512, help='Number of videos to generate')
    parser.add_argument('--frames_per_video', type=int, default=16, help='Frames per video after sampling')
    parser.add_argument('--frame_interval', type=int, default=3, help='Interval between frames')
    parser.add_argument('--fps', type=int, default=8, help='FPS for saved videos')
    parser.add_argument('--resolution', type=int, default=256, help='Resize resolution (assumes square)')

    args = parser.parse_args()
    save_videos_from_dataset(args)

'''

'''
import os
from glob import glob
from tqdm import tqdm
import random
import torch
import imageio
from torchvision import transforms
from torchvision.transforms import CenterCrop, ToTensor, Normalize
from PIL import Image

# ==== CONFIG ====
DATA_DIR = "/projects/SkyGAN/webcams/chmi.cz/sky_webcams" #"./datasets/sky_timelapse/sky_train"
SAVE_DIR = "/projects/tvdm/fvd_eval/real_videos/webcam/"
NUM_VIDEOS = 512
NUM_FRAMES = 16
FRAME_INTERVAL = 3
CROP_SIZE = 256
FPS = 8

# ==== TRANSFORM ====
transform = transforms.Compose([
    CenterCrop(CROP_SIZE),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ==== IO ====
os.makedirs(SAVE_DIR, exist_ok=True)
#video_dirs = sorted(glob(os.path.join(DATA_DIR, "*")))                        # SKy_timelapase

#video_dirs = sorted(glob(os.path.join(DATA_DIR, "*", "*")))                    # Webcam
video_dirs = glob(os.path.join(DATA_DIR, "*/*"))  # all <region>/<date> folders
random.shuffle(video_dirs)

saved_count = 0
# ==== PROCESS EACH VIDEO ====
for folder in tqdm(video_dirs, desc="Generating videos"):
    # List of frames inside subfolder e.g., video_name/video_name_1/*.jpg
    #all_images = sorted(glob(os.path.join(folder, "*/*.jpg")))                # sky_timelapse
    
    all_images = sorted(glob(os.path.join(folder, "*.jpg")))                #Webcam
    
    if len(all_images) < NUM_FRAMES * FRAME_INTERVAL:
        print(f"[Skip] Not enough frames in {folder}: {len(all_images)}")
        continue
    #Fix for webcam data path
    ########
    # Random starting point so that we don’t go out of bounds
    max_start = len(all_images) - NUM_FRAMES * FRAME_INTERVAL
    if max_start < 0:
        continue

    start_idx = random.randint(0, max_start)
    selected_images = [all_images[start_idx + i * FRAME_INTERVAL] for i in range(NUM_FRAMES)]

    ##########################

    selected_images = [all_images[i * FRAME_INTERVAL] for i in range(NUM_FRAMES)]
    frames = []
    for path in selected_images:
        img = Image.open(path).convert("RGB")
        tensor_img = transform(img)
        frames.append(tensor_img)

    video_tensor = torch.stack(frames)  # (T, C, H, W)
    video_tensor = ((video_tensor * 0.5 + 0.5) * 255).clamp(0, 255).to(torch.uint8)


    ################################ eliminating dark frames
    # Check brightness to avoid night/dark frames
    mean_brightness = video_tensor.float().mean().item() / 255.0
    if mean_brightness < 0.20:
        print(f"[Skip] Too dark — brightness={mean_brightness:.3f}")
        continue
    #################################

    video_np = video_tensor.permute(0, 2, 3, 1).numpy()  # (T, H, W, C)

    save_path = os.path.join(SAVE_DIR, f"{saved_count:04d}.mp4")
    imageio.mimwrite(save_path, video_np, fps=FPS, quality=9)
    saved_count += 1

    if saved_count >= NUM_VIDEOS:
        break

print(f"✅ Saved {saved_count} real videos to: {SAVE_DIR}")

'''

#!/usr/bin/env python
# real-video.py — Generate bright real webcam videos for FVD

import os, random
from glob import glob
from tqdm import tqdm
import torch, imageio
from torchvision import transforms
from torchvision.transforms import CenterCrop, ToTensor, Normalize
from PIL import Image

# -------------------------------------------------------------------------
#  Fixed Configuration (you can edit these if needed)
# -------------------------------------------------------------------------
DATA_ROOT   = "/projects/SkyGAN/webcams/chmi.cz/sky_webcams/"
SAVE_DIR    = "/projects/tvdm/fvd_eval/real_videos/webcam_0.18_full/"
NUM_VIDEOS  = 512
NUM_FRAMES  = 16
INTERVAL    = 1
CROP_SIZE   = 256
FPS         = 8
BRIGHT_THR  = 0.18  # skip dark clips

# -------------------------------------------------------------------------
#  TorchVision transform (crop + to tensor + normalize)
# -------------------------------------------------------------------------
transform = transforms.Compose([
    #CenterCrop(CROP_SIZE),
    transforms.Resize((CROP_SIZE, CROP_SIZE)),
    ToTensor(),
    Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# -------------------------------------------------------------------------
#  Start clip generation
# -------------------------------------------------------------------------
os.makedirs(SAVE_DIR, exist_ok=True)
folders = glob(os.path.join(DATA_ROOT, "*", "*"))  # all <region>/<date>
#folders = glob(os.path.join(DATA_ROOT, "*"))  # specific region/<date>
random.shuffle(folders)

print(f"🔎 Scanning {len(folders)} date folders "
      f"for ≥{NUM_FRAMES*INTERVAL} frames each…")

saved = 0
for folder in tqdm(folders, desc="Generating clips"):
    image_list = sorted(glob(os.path.join(folder, "*.jpg")))
    if len(image_list) < NUM_FRAMES * INTERVAL:
        continue

    # Random frame window
    max_start = len(image_list) - NUM_FRAMES * INTERVAL
    start = 0 if max_start <= 0 else random.randint(0, max_start)
    selected = [image_list[start + i * INTERVAL] for i in range(NUM_FRAMES)]

    # Load and transform
    try:
        frames = [transform(Image.open(p).convert("RGB")) for p in selected]
    except Exception as e:
        print(f"[Skip] Failed to load image: {e}")
        continue

    video_tensor = torch.stack(frames)                     # (T,C,H,W)

    # Brightness check
    mean_brightness = (video_tensor * 0.5 + 0.5).mean().item()
    if mean_brightness < BRIGHT_THR:
        print(f"[Skip] Too dark — brightness={mean_brightness:.3f}")
        continue

    # Save to mp4
    video_uint8 = ((video_tensor * 0.5 + 0.5) * 255).clamp(0, 255).byte()
    video_np = video_uint8.permute(0, 2, 3, 1).cpu().numpy()  # T,H,W,C

    save_path = os.path.join(SAVE_DIR, f"{saved:04d}.mp4")
    imageio.mimwrite(save_path, video_np, fps=FPS, quality=8)
    saved += 1

    if saved % 25 == 0 or saved == NUM_VIDEOS:
        print(f"[{saved}/{NUM_VIDEOS}] saved {save_path}")
    if saved >= NUM_VIDEOS:
        break

print(f"\n✅ Done: {saved} real bright clips saved to {SAVE_DIR}")
