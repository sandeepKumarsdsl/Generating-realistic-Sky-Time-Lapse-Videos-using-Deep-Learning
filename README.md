# Experiments Documentation

This directory contains scripts for running experiments.
Additionally, there are saved data in the subdirectories.

Python is required to run these scripts.
Please install it before running experiments.
The scripts also use additional packages listed in [requirements](requirements.txt).

## Creating Training Data 

# Sky_Timelapse Data

Collected from [1000] long timelapse videos from [Youtube]. Manually cut out the short videos with sky scenes from each long video.
Each short video contains multiple frames( from 2 frames to over 100 frames), each frame is a `640 * 360` resolution RGB image. 

The folder [sky_timelapse] contains the following folder and files:

`./sky_train`: training data of SkyScene. 
	contains 997 long timelapse videos, which are cut into 2392 short videos. 
	if we divide the short videos into clips of 32 frames, we can get 35383 video clips. 

`./sky_test`: test data of SkyScene. 
	contains 111 long timelapse videos, which are cut into 225 short videos. 
	note that the long videos in the test dataset have no overlap with the long videos in the training dataset.  
	if we divide the short videos into clips of 32 frames, we can get 2815 video clips. 

@InProceedings{Xiong_2018_CVPR,
author = {Xiong, Wei and Luo, Wenhan and Ma, Lin and Liu, Wei and Luo, Jiebo},
title = {Learning to Generate Time-Lapse Videos Using Multi-Stage Dynamic Generative Adversarial Networks},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}

# Webcam Data From CHMI

Dataset of weather webcam images scraped from the Czech Hydrometeorological institute [CHMI] over the past `three` years, on a `five` minutes interval.
Contains images captured all throughout the years from different geographical regions of `Czechia`.
It captures all the weather changes in the region `24/7` 

- Contains night, day, sunny, midday, foggy, mist, snow, rain, 
- [image-size] is `1600 * 1200` pixel resolution
- contains `31,927,520` video clips of `16` frames.

# Data structure

Datastructure describes how the two different datasets structures and stored into data frames.

/projects/
├── tvdm/
│   └── datasets/
│       └── sky_timelapse/
│           ├── sky_train/
│           │   └── [video_id]/[segment]/[frame_XXXXX.jpg]
│           ├── sky_val/
│           └── sky_test/
└── SkyGAN/
    └── webcams/
        └── chmi.cz/
            └── sky_webcams/
                └── [location]/[yyyymmdd]/[hhmm.jpg]


### Preparing Input Data

The collected data need to be modified for them to be used as input to 
For this purpose, a script is provided in the repository that can convert plans into the required formats.

The script [sky_image_datasets.py](sky_image_datasets.py), runs during training for dynamic preprocessing of the frames.
Preprocessed data is then feeded into the network for training.

The script [real-video.py](real-video.py). Generated real-video from the dataframes.
The frames are preprocessed and converted to a video with the frame interval of `3`.
Randomly selects the folders that have enough frame `>45`.

``` sh
python real-video.py
```
- Inputs - Path to the input directory and number of videos to be generated.
# Training

The arguments of the script are:

The script [train](train.py), starts the training on the data provided with the [config](config.yaml) setup.
Can resume training from the ckpt saved, loading pretrained weights or start training from step `0`.
The script is run as:
``` sh
python train.py [--config CONFIG_PATH] 
```
The arguments of the script are:
- `--config`: YAML configuration file defining model architecture and training parameters. Default is `./config/sky_sample.yaml/` type `str`.



# Sampling

The script [sample_multiple](sample_multiple.py), based on the sampler chosen `DDIM` or `DDPM`
sample videos in .mp4 format.
User can specific `N` number of sample and method for inferencing.
The script is run as:
``` sh
python sample/sample_multiple.py [--config CONFIG_PATH] 
    [--ckpt CHECKPOINT_PATH] [--save_video_path SAVE_DIR] 
    [--num_samples N] [--use_fp16] 
    [--sample_method {ddpm,ddim}]
```
The arguments of the script are:
- `--config`: YAML configuration file defining model architecture and sampling parameters. Default is `./config/sky_sample.yaml/`.
- `--ckpt`: Path to the pre-trained model checkpoint file in `.pt` format. Default is `./checkpoints/sky_timelapse/skytimelapse.pt`.
- `--save_video_path`: Output directory where the generated video samples will be saved. Default is `./sample_videos/`.
- `--num_samples`: Number of video samples to generate. Default value is `1`.
- `--use_fp16`: Enables 16-bit floating point precision for faster and memory-efficient sampling. Default Flag is `None`.
- `--sample_method`: Sampling method to be used for sampling sampling videos in `50-1000` steps. Ideal step is `250`. Default method is `DDPM`.


# Evaluation
Evaluating FVD and average CLIP similarity with `512` real videos vs `512` fake videos generated from the checkpoints

The script [run_fvd_metrics.py](run_fvd_metrics.py)
Fréchet Video Distance - Lower the value better the outcome.
``` sh
python run_fvd_metrics.py  
```
- Input - It takes real-video path and fake-video path.
- Output - Computed the FVD over the input videos. Ideal evaluation for videos

The script [framewise_CLIP_Score.py](framewise_CLIP_Score.py)
Compares the cosine similarity within th videos and averages it.
``` sh
python framewise_CLIP_Score.py
```
- Input - It takes real-video path and fake-video path.
- Output - Calculates the Average CLIP similarity and generated `.csv` file.

## Running GIMM-VFI 

FlexFringe can be downloaded or cloned from
[https://github.com/GSeanCDAT/GIMM-VFI](https://github.com/GSeanCDAT/GIMM-VFI).
There are predefined configuration INI file and also detailed documentation on this page.

The program can be run directly with

The script [video_Nx.py](video_Nx.py) is used sample interpolated video by adding flow and extending frames.
User can specific `N` number of frames to be added.
The script is run as:
``` sh
python src/video_Nx.py [--source-path SOURCE_DIR] 
    [--output-path OUTPUT_DIR] [--ds-factor DOWNSCALE] 
    [--N NUM_INTERPOLATED_FRAMES] [-m MODEL_CONFIG] 
    [-l CHECKPOINT_PATH] [--eval]
```
The arguments of the script are:
- `--source-path`: Path to the input directory containing videos or image sequences to be interpolated. Default is `./inputs/`.
- `--output-path`: Directory where the interpolated output videos or frames will be saved.. Default is `./outputs/interpolated/`.
- `--ds-factor`: Downscaling factor for input resolution during processing. A value of `1` means no downscaling. Default value is `1`.
- `--N`: The number of frames to interpolate between each pair of input frames. Default value is `1`. `2` will generate two intermediate frames.
- `--m`: Path to the model configuration YAML file for the GIMMVFI model. Default is `./configs/gimmvfi/gimmvfi_r_arb.yaml`.
- `--l`: Path to the pre-trained model checkpoint `.pt` to be loaded for inference.
- `--eval`: Runs the script in evaluation mode, typically used for inference.


# Tensorboard check 

tensorboard --logdir /path/to/checkpoint --port 6006 --bind_all -- To be intiated within the cluster

ssh -L 6006:localhost:6006 username@servername  -- To be launched in the terminal

[https://localhost:6006](https://localhost:6006) - locally access URL to view the events.
