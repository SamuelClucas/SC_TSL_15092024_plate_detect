"""
Custom Dataset class for loading and processing plate images with bounding box annotations.

This class implements a map-style dataset that provides indexed access to image data
and their corresponding bounding box annotations. It inherits from torch.utils.data.Dataset
and is designed to work with PyTorch's DataLoader.

References:
    - PyTorch Dataset Tutorial: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    - Dataset Documentation: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
    - Map-style datasets: https://pytorch.org/docs/stable/data.html#map-style-datasets
"""

import os
import re
from glob import glob
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.v2 import functional as F
from torchvision import tv_tensors


class Plate_Image_Dataset(Dataset):
    """
    Dataset class for loading and processing images with plate annotations.
    
    Attributes:
        img_labels (pd.DataFrame): DataFrame containing image annotations
        img_files (List[str]): List of paths to image files
        transforms (Optional[Any]): Optional transforms to be applied to images
    """

    def __init__(
        self, 
        annotations_file: str, 
        img_dir: str, 
        transforms: Optional[Any] = None
    ) -> None:
        """
        Initialize the dataset.

        Args:
            annotations_file: Path to CSV file containing image annotations
            img_dir: Directory containing the image files
            transforms: Optional transforms to be applied to images and targets
        
        Raises:
            FileNotFoundError: If annotations_file or img_dir doesn't exist
            pd.errors.EmptyDataError: If annotations file is empty
        """
        if not os.path.exists(annotations_file):
            raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
            
        self.img_labels = pd.read_csv(annotations_file)
        self.img_files = sorted(glob(os.path.join(img_dir, "*.png")))
        self.transforms = transforms

    def __len__(self) -> int:
        """Return the total number of images in the dataset."""
        return len(self.img_files)
    
    def load_image(self, idx: int) -> torch.Tensor:
        """
        Load a single image without applying transforms.
        
        Args:
            idx: Index of the image to load
            
        Returns:
            Raw image tensor
            
        Note:
            This method preserves original image values for visualization with matplotlib
        """
        return read_image(self.img_files[idx])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Get an image and its annotations by index.
        
        Args:
            idx: Index of the image to retrieve
            
        Returns:
            Tuple containing:
                - Image tensor (transformed if transforms are specified)
                - Dictionary containing:
                    - boxes: Bounding box coordinates in XYXY format
                    - labels: Object class labels
                    - image_id: Index of the image
                    - area: Area of each bounding box
                    - iscrowd: Crowd annotation flags
                    
        Raises:
            IndexError: If idx is out of bounds
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
            
        # Load and convert image to TVTensor
        img = tv_tensors.Image(read_image(self.img_files[idx]))
        
        # Extract bounding box coordinates from annotations
        x1 = [int(x) for x in re.findall(r'\d+', str(self.img_labels['xmins'][idx]))]
        y1 = [int(y) for y in re.findall(r'\d+', str(self.img_labels['ymins'][idx]))]
        x2 = [int(x) for x in re.findall(r'\d+', str(self.img_labels['xmaxs'][idx]))]
        y2 = [int(y) for y in re.findall(r'\d+', str(self.img_labels['ymaxs'][idx]))]
        
        # Create bounding boxes tensor
        num_objs = len(x1)
        boxes = torch.tensor(
            [[x1[i], y1[i], x2[i], y2[i]] for i in range(num_objs)],
            dtype=torch.float32
        )
        
        # Create target dictionary with all required annotations
        target = {
            "boxes": tv_tensors.BoundingBoxes(
                boxes,
                format="XYXY",
                canvas_size=F.get_size(img)
            ),
            "labels": torch.ones((num_objs,), dtype=torch.int64),  # All objects are class 1 (plate)
            "image_id": idx,
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),  # height * width
            "iscrowd": torch.zeros((num_objs,), dtype=torch.int64)  # No crowd annotations
        }

        # Apply transforms if specified
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target