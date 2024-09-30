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

class Negatives_Image_Dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir: str, transforms=None):

        self.boxes: np.array[int] = np.asarray(torch.zeros((0, 4), dtype=torch.float32))

        self.img_files: list[str] = [f for f in glob(img_dir + "/*.png")]
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.img_files)
    
    def __getitem__(self, idx: int) -> Dict[torch.Tensor, tv_tensors.Image]: 
        # loads images and bounding boxes
        img: torch.Tensor = read_image(self.img_files[idx]) # uint8
        
        num_objs = 0
        labels = torch.ones((num_objs,), dtype=torch.int64) #int64 could be overkill, int8 since only one label, maybe downstream methods expect type   int64 so using for now as specified here: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
        # one BoundingBoxes instance per sample, "{img": img, "bbox": BoundingBoxes(...)}" where BoundingBoxes contains all the bounding box vertices associated with that image in the form x1, y1,x2, y2
        self.boxes = torch.from_numpy(np.array(self.boxes)) # [N, 4] tensor, m x n matrix , Rows by cols
        print(self.boxes.shape)
        image_id = idx
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        area = 0
        # tv.tensors is tensor of images with associated metadata
        img = tv_tensors.Image(img)
        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(self.boxes, format="XYXY", canvas_size=F.get_size(img))
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        #print(f'area is: {area}')
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, target
