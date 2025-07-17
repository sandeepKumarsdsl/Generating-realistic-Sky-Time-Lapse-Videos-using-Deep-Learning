import os
from torch.utils.data import Dataset
import torchvision.io as io

class VideoFolderDataset(Dataset):
    def __init__(self, path, resolution=256, use_labels=False, load_n_consecutive=16, **kwargs):
        self.path = path
        self.videos = sorted([
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith('.mp4')
        ])
        self.resolution = resolution
        self.load_n_consecutive = load_n_consecutive
        self.use_labels = use_labels
        # We just ignore **kwargs like max_size, xflip, etc.

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        video, _, _ = io.read_video(video_path, pts_unit='sec')  # [T, H, W, C]
        video = video.permute(0, 3, 1, 2)  # [T, C, H, W]
        video = video[:self.load_n_consecutive]
        return {'image': video}
