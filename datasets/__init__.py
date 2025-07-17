
from torchvision import transforms
from datasets import video_transforms
from .sky_image_datasets import SkyImages



def get_dataset(args):
    temporal_sample = video_transforms.TemporalRandomCrop(args.num_frames * args.frame_interval) # 16 1
    '''
    if args.dataset == 'ffs':
        transform_ffs = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(args.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        return FaceForensics(args, transform=transform_ffs, temporal_sample=temporal_sample)
    elif args.dataset == 'ffs_img':
        transform_ffs = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(args.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        return FaceForensicsImages(args, transform=transform_ffs, temporal_sample=temporal_sample)
    elif args.dataset == 'ucf101':
        transform_ucf101 = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(args.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        return UCF101(args, transform=transform_ucf101, temporal_sample=temporal_sample)
    elif args.dataset == 'ucf101_img':
        transform_ucf101 = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(args.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        return UCF101Images(args, transform=transform_ucf101, temporal_sample=temporal_sample)
    elif args.dataset == 'taichi':
        transform_taichi = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        return Taichi(args, transform=transform_taichi, temporal_sample=temporal_sample)
    elif args.dataset == 'taichi_img':
        transform_taichi = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        return TaichiImages(args, transform=transform_taichi, temporal_sample=temporal_sample)
    elif args.dataset == 'sky':  
        transform_sky = transforms.Compose([
                    video_transforms.ToTensorVideo(),
                    video_transforms.CenterCropResizeVideo(args.image_size),
                    # video_transforms.RandomHorizontalFlipVideo(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        return Sky(args, transform=transform_sky, temporal_sample=temporal_sample)
    '''
    if args.dataset == 'sky_img':  
        transform_sky = transforms.Compose([
                    video_transforms.ToTensorVideo(),
                    video_transforms.CenterCropResizeVideo(args.image_size),
                    # video_transforms.RandomHorizontalFlipVideo(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        return SkyImages(args, transform=transform_sky, temporal_sample=temporal_sample)
    else:
        raise NotImplementedError(args.dataset)