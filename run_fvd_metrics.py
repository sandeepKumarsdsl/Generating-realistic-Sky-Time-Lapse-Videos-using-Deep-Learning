import os
import sys
import torch
from types import SimpleNamespace

# Add tools directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "tools"))

from tools.metrics import metric_main

# Metric options for real/fake video folders
opts = {
    "dataset_kwargs": {
        "class_name": "tools.dataset.video_folder_dataset.VideoFolderDataset",
        "path": "./fvd_eval/real_videos/webcam_0.20_mean/",
        "resolution": 256,
        "use_labels": False,
        "load_n_consecutive": 16,
    },
    "gen_dataset_kwargs": {
        "class_name": "tools.dataset.video_folder_dataset.VideoFolderDataset",
        "path": "./fvd_eval/fake_videos/webcam_ddim_220k/",
        "resolution": 256,
        "use_labels": False,
        "load_n_consecutive": 16,
    },
    "generator_as_dataset": True,
    "num_gpus": 1,
    "rank": 0,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "cache": False,
}

# Run the metric
results = metric_main.calc_metric(metric="fvd2048_16f", **opts)
print(f"\n🎯 FVD Score: {results['results'].fvd2048_16f:.4f}")
