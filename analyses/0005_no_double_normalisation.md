# Retraining without Redundant Normalisation

2/11/25

I realised the normalisation transform I have been applying is
redundant, and it may well be affecting performance.

### Imports:

``` python
from plate_detect import helper_training_functions, Plate_Image_Dataset
import torchvision_deps
from torchvision.ops.boxes import masks_to_boxes
import os
import numpy as np
import pandas as pd
from torchvision.io import read_image
import torch
from torchvision.transforms.v2 import functional as F
from torchvision import tv_tensors
from typing import Dict
import torchvision_deps.T_and_utils
from collections import defaultdict
from pathlib import Path
from torchvision import transforms
```

The model already applies a normalisation transform on pixel intensities
of the input samples by default. I don’t need to carry out a
normalisation transform, just a datatype conversion to float32 to the
Plate_Image_Dataset constructor.

``` python
import caffeine

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

project_dir_root: Path= Path.cwd() # 'SC_TSL_15092024_Plate_Detect/' type PosixPath for UNIX, WindowsPath for windows...
print(f'Project root directory: {str(project_dir_root)}')

annotations_file: Path = project_dir_root.parents[0].joinpath('lib', 'labels.csv')
print(f'Training labels csv file: {annotations_file}')

img_dir: Path= project_dir_root.parents[0].joinpath('raw', 'positives')  # 'SC_TSL_15092024_Plate_Detect/train/images/positives/' on UNIX systems
print(f'Training dataset directory: {img_dir}')

transform = transforms.v2.Compose([
    #transforms.v2.ToImage(),
    transforms.v2.ToDtype(torch.float32, scale=True)
])

# Setup data
dataset = Plate_Image_Dataset.Plate_Image_Dataset(
    annotations_file=str(annotations_file),
    img_dir=str(img_dir),
    transforms=transform  # No double transform!
)

# Split dataset with correct boxes now
dataset_size = len(dataset)
validation_size = 50  # Or whatever size you want
indices = [int(i) for i in torch.randperm(dataset_size).tolist()]

dataset_validation = torch.utils.data.Subset(dataset, indices[-validation_size:])
dataset_train = torch.utils.data.Subset(dataset, indices[:-validation_size])

# Create data loaders with custom_collate_fn
train_loader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=1,
    shuffle=True,
    collate_fn=torchvision_deps.T_and_utils.utils.collate_fn
)

validation_loader = torch.utils.data.DataLoader(
    dataset_validation,
    batch_size=1,
    shuffle=False,
    collate_fn=torchvision_deps.T_and_utils.utils.collate_fn
)

# Rest of your training setup
model, optimizer, preprocess = helper_training_functions.get_model_instance_object_detection(num_class=2)
model.to(device)

# Training loop
num_epochs = 15
precedent_epoch = 0
save_dir = '../'

epoch = helper_training_functions.train(model, train_loader, validation_loader, device, num_epochs, precedent_epoch, save_dir, optimizer)
```

Results show improved performance when compared to the evaluation
results from [analysis 0002](0002_functional_Faster_R-CNN.md).

**Analysis 0002:**

![evaluation_analysis_0002](../results/0002_15_epochs/02_fix/evaluation_metrics_epochs_0-15.png)

**Analysis 0005:**

![evaluation_analysis_0005](../results/0005_no_double_normalisation/evaluation_metrics_epochs_0-15.png)
