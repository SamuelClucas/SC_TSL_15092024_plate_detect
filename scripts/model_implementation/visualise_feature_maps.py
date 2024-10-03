"""
Implementation of a 'Faster R-CNN' (as here: https://arxiv.org/pdf/1506.01497).
    Trained on: 'train/images/positives/'

    Before running this script:
        - Run 'pip install -e .' so that custom imports from ./src/ work 

Custom imports from src:
- Plate_Image_Dataset.py, class Plate_Image_Dataset to load and provide easy access to images.

"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from src import Plate_Image_Dataset, helper_training_functions
import torch
# torchvision.transforms.v2 now recommend (faster, do more) (see here:https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.ToTensor)
from torchvision.transforms.v2 import functional as F
from pathlib import Path
from typing import Dict, Union
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2, FastRCNNPredictor, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms import v2 as T
import lib.utils as utils
from lib.engine import train_one_epoch, evaluate 
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import PIL
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
from torch import nn

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

"""
Firstly, load the training dataset by instantiating  object of custom type Plate_Image_Dataset (see documentation here: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).
- the data and its labels must be transformed so that they're suitable for training (https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html).
    - images as normalised tensors by passing torchvision.transforms.ToTensor() as class constructor transform argument.
    - labels as one-hot encoded tensors using torchvision.transforms.Lambda() as class constructor target_transform argument.
        - 'one-hot encoding': represent categorical data in a numerical format
"""

project_dir_root: Path= Path.cwd() # 'SC_TSL_15092024_Plate_Detect/' type PosixPath for UNIX, WindowsPath for windows...
print(f'Project root directory: {str(project_dir_root)}')

annotations_file: Path = project_dir_root.joinpath('train', 'labels.csv')
print(f'Training labels csv file: {annotations_file}')

img_dir: Path= project_dir_root.joinpath('train', 'images', 'positives')  # 'SC_TSL_15092024_Plate_Detect/train/images/positives/' on UNIX systems
print(f'Training dataset directory: {img_dir}')

num_class = 2 # plate or background
# creates resnet50 v2 faster r cnn model with new head for class classification
model, preprocess = helper_training_functions.get_model_instance_object_detection(num_class)
# move model to the right device
model.to(device)

dataset: Plate_Image_Dataset = Plate_Image_Dataset.Plate_Image_Dataset(
    img_dir=str(img_dir), 
    annotations_file=str(annotations_file),
    transforms=preprocess, # converts Tensor image, PIL image, NumPy ndarray into FloatTensor and scales pixel intensities in range [0.,1.].
    )

# split the dataset in train and test set
dataset_size = len(dataset)
test_size = min(50, int(dataset_size // 5))  # Use 20% of data for testing, or 50 samples, whichever is smaller
indices = [int(i) for i in torch.randperm(dataset_size).tolist()]
dataset_test = torch.utils.data.Subset(dataset, indices[-test_size:])
dataset = torch.utils.data.Subset(dataset, indices[:-test_size])

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    collate_fn=utils.collate_fn
)

img, otherthing = dataset[54]
num_epochs = 1
print_freq = 10
save_dir = './checkpoints/'  

model = helper_training_functions.load_model(model, save_dir)
with torch.no_grad():
    image = img
    image = image[:3, ...].to(device)
    predictions = model([image, ])
    pred = predictions[0]

#image = 255.0 * (image - image.min() / (image.max() - image.min())) - image.min()
image = image[:3, ...]
pred_boxes = pred["boxes"].long()
output_image = draw_bounding_boxes(image, pred_boxes, colors="red")

plt.figure(figsize=(12, 12))
plt.imshow(output_image.permute(1, 2, 0))
plt.show()

print(f'\n-Plot feature maps-')

feature_maps = helper_training_functions.get_feature_maps(model, image)
helper_training_functions.visualise_feature_maps(feature_maps)


print("\n -end-")