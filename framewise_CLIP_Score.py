'''
import torch
import clip
import os
import cv2
from PIL import Image
from glob import glob
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def extract_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return img

def compute_clip_similarity(img1, img2):
    images = torch.stack([preprocess(img1), preprocess(img2)]).to(device)
    with torch.no_grad():
        features = model.encode_image(images)
        features = features / features.norm(dim=-1, keepdim=True)
        return features[0].dot(features[1]).item()  # cosine similarity

# Example usage
real_paths = sorted(glob("./fvd_eval/real_videos/sky_timelapse/*.mp4"))
fake_paths = sorted(glob("./fvd_eval/fake_videos/sky_timelapse_ddim_120k/*.mp4"))

scores = []
for real, fake in zip(real_paths, fake_paths):
    
    
    img_real = extract_first_frame(real)
    img_fake = extract_first_frame(fake)
    if img_real and img_fake:
        score = compute_clip_similarity(img_real, img_fake)
        scores.append(score)

print(f"\n🎯 Avg CLIP similarity (cosine): {sum(scores)/len(scores):.4f}")


'''

######   All frames  #############


import torch
import clip
import os
import cv2
import csv
from PIL import Image
from glob import glob
from torchvision.transforms import Compose
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def extract_frames(video_path, max_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sorted(set(torch.linspace(0, total_frames - 1, max_frames).long().tolist()))

    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(preprocess(img).unsqueeze(0))  # [1, 3, 224, 224]
    cap.release()
    return frames

def compute_clip_similarity_frames(frames1, frames2):
    n = min(len(frames1), len(frames2))
    if n == 0:
        return None

    scores = []
    for i in range(n):
        with torch.no_grad():
            f1 = frames1[i].to(device)
            f2 = frames2[i].to(device)
            feats = model.encode_image(torch.cat([f1, f2], dim=0))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            sim = feats[0].dot(feats[1]).item()
            scores.append(sim)
    return sum(scores) / len(scores)

# Load video paths
real_paths = sorted(glob("./fvd_eval/real_videos/sky_timelapse/*.mp4"))
fake_paths = sorted(glob("./fvd_eval/fake_videos/sky_timelapse_ddim_120k/*.mp4"))

csv_rows = []
scores = []

for real, fake in tqdm(zip(real_paths, fake_paths), total=len(real_paths)):
    real_name = os.path.basename(real)
    fake_name = os.path.basename(fake)

    real_frames = extract_frames(real)
    fake_frames = extract_frames(fake)

    sim = compute_clip_similarity_frames(real_frames, fake_frames)
    if sim is not None:
        scores.append(sim)
        csv_rows.append([real_name, fake_name, f"{sim:.4f}"])

# Save to CSV
csv_path = "clip_similarity_scores_sky_timelapse_120k.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Real Video", "Fake Video", "CLIP Similarity"])
    writer.writerows(csv_rows)

# Final output
if scores:
    avg_clip = sum(scores) / len(scores)
    print(f"\nAvg multi-frame CLIP similarity: {avg_clip:.4f}")
    print(f"Saved detailed scores to: {csv_path}")
else:
    print("❌ No valid video pairs found.")
