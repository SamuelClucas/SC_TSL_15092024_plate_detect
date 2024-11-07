"""
Custom class 'Plate_Image_Dataset' to load and provide easy access to images.

See pytorch datasets and dataloaders beginner tutorial here: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    - custom Dataset classes inherit from parent abstract class torch.utils.data.Dataset (https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset), where this child class is a map-style dataset (see here under dataset types: https://pytorch.org/docs/stable/data.html)
    - maps indices to data samples 
    - overwrites __getitem__() used to fetch sample based on a given key
    - optionally overwrite __len__() expected to return dataset size by DataLoader
    - NOT IMPLEMENTED HERE: optional __getitems__() recommended to speedup batched samples loading, accepting batch list of sample indices, returning list of samlples
"""
import os
import re
from glob import glob
import numpy as np
import pandas as pd
from torchvision.io import read_image
import torch
from torchvision.transforms.v2 import functional as F
from torchvision import tv_tensors
from typing import Dict

class Plate_Image_Dataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file: str, img_dir: str, transforms=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_files = [f for f in sorted(glob(img_dir + "/*.png"))]
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.img_files)
    
    def load_image(self, idx): # adding this as normalisation will mess up plotting with matplotlib
        print(self.img_files)
        img = read_image(self.img_files[idx])
        return img

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Dict]:
        # Load image 
        img = read_image(self.img_files[idx])
        # Convert image to TVTensor
        img = tv_tensors.Image(img)
        
        # Get coordinates
        x1 = [int(item) for item in re.findall(r'\d+', str(self.img_labels['xmins'][idx]))]
        y1 = [int(item) for item in re.findall(r'\d+', str(self.img_labels['ymins'][idx]))]
        x2 = [int(item) for item in re.findall(r'\d+', str(self.img_labels['xmaxs'][idx]))]
        y2 = [int(item) for item in re.findall(r'\d+', str(self.img_labels['ymaxs'][idx]))]
        
        # Create boxes tensor
        num_objs = len(x1)
        boxes = torch.tensor(
            [[x1[i], y1[i], x2[i], y2[i]] for i in range(num_objs)],
            dtype=torch.float32
        )
        
        # Create target dictionary
        target = {
            "boxes": tv_tensors.BoundingBoxes(
                boxes,
                format="XYXY",
                canvas_size=F.get_size(img)
            ),
            "labels": torch.ones((num_objs,), dtype=torch.int64),
            "image_id": idx,
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((num_objs,), dtype=torch.int64)
        }

        # Apply transforms only to img
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target