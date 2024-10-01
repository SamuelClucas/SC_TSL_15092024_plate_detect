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
    pass
        

if __name__ == "__main__":

    unittest.main(verbosity=2)