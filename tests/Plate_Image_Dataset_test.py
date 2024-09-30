"""
Each test case class inherits from unittest.TestCase class.

Use setupt and teardown features!!! https://realpython.com/python-unittest/#creating-test-fixtures

Each test method within this class represents a specific test scenario for Plate_Image_Dataset class defined in 'src/Plate_Image_Dataset.py'.
"""

import unittest
from src.Plate_Image_Dataset import Plate_Image_Dataset

import os
import re
from glob import glob
import numpy as np
import pandas as pd
from torchvision.io import read_image
import torch
from torchvision.transforms.v2 import functional as F
from typing import Dict
from pathlib import Path
    
class Test_Plate_Image_Dataset(unittest.TestCase):
    def test_annotations_file_is_equal_length_images_in_img_dir(self, ds.annotations_file: str, ds.img_files: list[str], img_dir: str):
    pass

    def main(self, dataset: Plate_Image_Dataset):
        project_dir_root: Path= Path.cwd() # 'SC_TSL_15092024_Plate_Detect/' type PosixPath for UNIX, WindowsPath for windows...
        print(f'Project root directory: {str(project_dir_root)}')

        annotations_file: Path = project_dir_root.joinpath('train', 'labels.csv')
        print(f'Training labels csv file: {annotations_file}')

        img_dir: Path= project_dir_root.joinpath('train', 'images', 'positives')  # 'SC_TSL_15092024_Plate_Detect/train/images/positives/' on UNIX systems
        print(f'Training dataset directory: {img_dir}')

        ds: Plate_Image_Dataset = Plate_Image_Dataset.Plate_Image_Dataset(
            img_dir=str(img_dir), 
            annotations_file=str(annotations_file),
            transforms=None, # converts Tensor image, PIL image, NumPy ndarray into FloatTensor and scales pixel intensities in range [0.,1.].
            )
        
        self.test_annotations_file_is_equal_length_images_in_img_dir(ds.img_labels, ds.img_files, img_dir)
        

if __name__ == "__main__":
    
    
    unittest.main(verbosity=2)