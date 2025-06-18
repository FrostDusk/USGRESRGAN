# train.py

import os
import os.path as osp

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

# Import necessary modules from basicsr
import torch

# These ensure custom modules (like archs, data, models) are registered correctly
import archs
import data
import losses
import models
from basicsr.train import train_pipeline

def main():
    root_path = osp.abspath(osp.dirname(__file__))  # Path to this script directory
    print(f"Starting training from root path: {root_path}")

    # Optional: check CUDA availability
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Training will be slow.")

    train_pipeline(root_path)

if __name__ == '__main__':
    main()
