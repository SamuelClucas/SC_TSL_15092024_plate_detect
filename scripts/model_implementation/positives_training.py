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

from src import Plate_Image_Dataset
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

def get_model_instance_bounding_boxes(num_class: int) -> fasterrcnn_resnet50_fpn_v2:
    # New weights with accuracy 80.858%
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT # alias is .DEFAULT suffix, weights = None is random initialisation, box MAP 46.7, params, 43.7M, GFLOPS 280.37 https://github.com/pytorch/vision/pull/5763
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.001)
    preprocess = weights.transforms()
    # finetuning pretrained model by transfer learning
    # get num of input features for classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_class)
    return model, preprocess

def save_checkpoint(model, optimizer, epoch, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    filename = f'checkpoint_epoch_{epoch}.pth'
    save_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")

def train(dataset, model):
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )
    for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

        save_checkpoint(model, optimizer, epoch, save_dir)
    # update the learning rate
        lr_scheduler.step()
    # evaluate on the test dataset
        model.eval()
    evaluate(model, data_loader_test, device=device)
    return model, optimizer, epoch

def load_model(model, save_dir):
    checkpoint = torch.load(save_dir + 'checkpoint_epoch_0.pth', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch']
    model.eval()
    return model

def get_feature_maps(model, input_image):
    feature_maps = {}
    
    def hook_fn(module, input, output):
        feature_maps[module] = output
    
    # Register hooks for the layers you want to visualize
    # Here we're registering hooks for all convolutional layers in the backbone
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.Conv2d):
            module.register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        model([input_image])
    
    return feature_maps

def visualize_feature_maps(feature_maps, num_features=64):
    for layer, feature_map in feature_maps.items():
        # Get the first image in the batch
        feature_map = feature_map[0]
        
        # Plot up to num_features feature maps
        num_features = min(feature_map.size(0), num_features)
        
        fig, axs = plt.subplots(8, 8, figsize=(20, 20))
        fig.suptitle(f'Feature Maps for Layer: {layer}')
        
        for i in range(num_features):
            ax = axs[i // 8, i % 8]
            ax.imshow(feature_map[i].cpu(), cmap='gray')
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

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
model, preprocess = get_model_instance_bounding_boxes(num_class)
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

model = load_model(model, save_dir)
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

feature_maps = get_feature_maps(model, image)
visualize_feature_maps(feature_maps)


print("\n -end-")