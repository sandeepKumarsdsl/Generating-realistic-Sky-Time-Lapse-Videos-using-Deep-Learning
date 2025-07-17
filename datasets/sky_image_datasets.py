'''

import os
import torch
import random
import torch.utils.data as data
import numpy as np
import copy
from PIL import Image

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class SkyImages(data.Dataset):
    def __init__(self, configs, transform, temporal_sample=None, train=True):

        self.configs = configs
        self.data_path = configs.data_path
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.target_video_len = self.configs.num_frames
        self.frame_interval = self.configs.frame_interval
        self.data_all, self.video_frame_all = self.load_video_frames(self.data_path)

        # sky video frames
        random.shuffle(self.video_frame_all)
        self.use_image_num = configs.use_image_num

    def __getitem__(self, index):
        
        video_index = index % self.video_num
        vframes = self.data_all[video_index]
        total_frames = len(vframes)

        # Sampling video frames
        start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        assert end_frame_ind - start_frame_ind >= self.target_video_len
        frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, num=self.target_video_len, dtype=int) # start, stop, num=50

        select_video_frames = vframes[frame_indice[0]: frame_indice[-1]+1: self.frame_interval] 
        
        video_frames = []
        for path in select_video_frames:
            #print(f"Trying to open video frame: {path}") # ADD THIS LINE
            video_frame = torch.as_tensor(np.array(Image.open(path), dtype=np.uint8, copy=True)).unsqueeze(0)
            video_frames.append(video_frame)
        video_clip = torch.cat(video_frames, dim=0).permute(0, 3, 1, 2)
        video_clip = self.transform(video_clip)

        # get video frames
        images = []

        for i in range(self.use_image_num):
            while True:
                try:      
                    video_frame_path = self.video_frame_all[index+i]
                    #print(f"Trying to open ADDITIONAL image: {video_frame_path=}") # MODIFIED PRINT STATEMENT
                    image = torch.as_tensor(np.array(Image.open(video_frame_path), dtype=np.uint8, copy=True)).unsqueeze(0)
                    images.append(image)
                    break
                except Exception as e:
                    index = random.randint(0, self.video_frame_num - self.use_image_num)

        images =  torch.cat(images, dim=0).permute(0, 3, 1, 2)
        images = self.transform(images)
        assert len(images) == self.use_image_num

        video_cat = torch.cat([video_clip, images], dim=0)

        return {'video': video_cat, 'video_name': 1}

        
    def __len__(self):
        return self.video_frame_num
    
    def load_video_frames(self, dataroot):
        data_all = []
        frames_all = []
        frame_list = os.walk(dataroot)
        for _, meta in enumerate(frame_list):
            root = meta[0]
            try:
                frames = sorted(meta[2], key=lambda item: int(item.split('.')[0].split('_')[-1]))
            except:
                print(meta[0]) # root
                print(meta[2]) # files
            frames = [os.path.join(root, item) for item in frames if is_image_file(item)]
            if len(frames) > max(0, self.target_video_len * self.frame_interval): # need all > (16 * frame-interval) videos
            # if len(frames) >= max(0, self.target_video_len): # need all > 16 frames videos
                data_all.append(frames)
                for frame in frames:
                    frames_all.append(frame)
        self.video_num = len(data_all)
        self.video_frame_num = len(frames_all)
        return data_all, frames_all
     '''

import os
import torch
import random
import torch.utils.data as data
import numpy as np
from PIL import Image


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG'
]

def is_image_file(filename: str) -> bool:
    return any(filename.endswith(ext) for ext in IMG_EXTENSIONS)

class SkyImages(data.Dataset):
    """SkyTimelapse / Webcam image‑folder dataset with fixed‑size output."""

    def __init__(self, configs, transform, temporal_sample=None, train=True):
        self.configs          = configs
        self.data_path        = configs.data_path
        self.transform        = transform              # expects 4‑D (T,C,H,W)
        self.temporal_sample  = temporal_sample
        self.target_video_len = configs.num_frames
        self.frame_interval   = configs.frame_interval
        self.resize_side      = configs.image_size if hasattr(configs, 'image_size') else 256

        # Collect frame paths
        self.data_all, self.video_frame_all = self._load_video_frames(self.data_path)
        random.shuffle(self.video_frame_all)
        self.use_image_num = configs.use_image_num

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------
    def __getitem__(self, index):
        video_index   = index % self.video_num
        vframes       = self.data_all[video_index]
        total_frames  = len(vframes)

        # Temporal sampling --------------------------------------------------
        start_f, end_f = self.temporal_sample(total_frames)
        assert end_f - start_f >= self.target_video_len
        frame_idx = np.linspace(start_f, end_f-1, num=self.target_video_len, dtype=int)
        selected   = vframes[frame_idx[0] : frame_idx[-1]+1 : self.frame_interval]

        # ------------ Load & resize each frame to fixed size ----------------
        video_frames = []
        for p in selected:
            try:
                img = Image.open(p).convert('RGB').resize((self.resize_side, self.resize_side), Image.BICUBIC)
                arr = np.asarray(img, dtype=np.uint8)              # (H,W,C)
                video_frames.append(torch.from_numpy(arr).unsqueeze(0))  # (1,H,W,C)
            except Exception as e:
                print(f"[Warning] failed to load frame {p}: {e}")

        if len(video_frames) == 0:
            raise RuntimeError("No valid frames in sampled clip; retry.")

        video_clip = torch.cat(video_frames, dim=0).permute(0,3,1,2)  # (T,C,H,W)
        video_clip = self.transform(video_clip)  # ToTensorVideo+crop+norm

        # ---------------------- Additional context images -------------------
        context = []
        for i in range(self.use_image_num):
            retries = 0
            while retries < 5:
                try:
                    img_path = self.video_frame_all[(index + i) % self.video_frame_num]
                    img = Image.open(img_path).convert('RGB').resize((self.resize_side, self.resize_side), Image.BICUBIC)
                    arr = np.asarray(img, dtype=np.uint8)
                    context.append(torch.from_numpy(arr).unsqueeze(0))
                    break
                except Exception as e:
                    retries += 1
                    index = random.randint(0, self.video_frame_num - self.use_image_num)
                    print(f"[Warning] retry loading context img {img_path}: {e}")

        context = torch.cat(context, dim=0).permute(0,3,1,2)
        context = self.transform(context)
        assert len(context) == self.use_image_num

        # Concatenate video frames + context images along time dimension
        video_cat = torch.cat([video_clip, context], dim=0)  # (T+N, C, H, W)
        return {'video': video_cat, 'video_name': 1}

    # ------------------------------------------------------------------
    # Length helpers
    # ------------------------------------------------------------------
    def __len__(self):
        return self.video_frame_num

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _load_video_frames(self, root_dir):
        """Walk the directory and collect frames for each clip."""
        data_all, frames_all = [], []
        for dirpath, _, filenames in os.walk(root_dir):
            try:
                filenames = sorted(filenames, key=lambda x: int(x.split('.')[0].split('_')[-1]))
            except ValueError:
                continue  # non‑numeric frame names

            frame_paths = [os.path.join(dirpath, f) for f in filenames if is_image_file(f)]
            if len(frame_paths) > self.target_video_len * self.frame_interval:
                data_all.append(frame_paths)
                frames_all.extend(frame_paths)

        self.video_num        = len(data_all)
        self.video_frame_num  = len(frames_all)
        return data_all, frames_all



if __name__ == '__main__':

    import argparse
    import torchvision
    import video_transforms
    import torch.utils.data as data

    from torchvision import transforms
    from torchvision.utils import save_image


    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_interval", type=int, default=3)
    parser.add_argument("--data-path", type=str, default="/path/to/datasets/sky_timelapse/sky_train/")
    parser.add_argument("--use-image-num", type=int, default=5)
    config = parser.parse_args()

    target_video_len = config.num_frames

    temporal_sample = video_transforms.TemporalRandomCrop(target_video_len * config.frame_interval)
    trans = transforms.Compose([
        video_transforms.ToTensorVideo(),
        # video_transforms.CenterCropVideo(256),
        video_transforms.CenterCropResizeVideo(256),
        # video_transforms.RandomHorizontalFlipVideo(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    sky_dataset = SkyImages(config, transform=trans, temporal_sample=temporal_sample)
    print(len(sky_dataset))
    sky_dataloader = data.DataLoader(dataset=sky_dataset, batch_size=1, shuffle=False, num_workers=1)

    for i, video_data in enumerate(sky_dataloader):
        print(video_data['video'].shape)
        
        # print(video_data.dtype)
        # for i in range(target_video_len):
        #     save_image(video_data[0][i], os.path.join('./test_data', '%04d.png' % i), normalize=True, value_range=(-1, 1))

        # video_ = ((video_data[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1)
        # torchvision.io.write_video('./test_data' + 'test.mp4', video_, fps=8)
        # exit()