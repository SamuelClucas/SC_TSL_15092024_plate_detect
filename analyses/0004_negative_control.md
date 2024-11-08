# Implementing a Negative Control

7/11/24

## Introduction

To complement the positive control from [analysis
0003](0003_positive_control.md), I will use the same model I trained in
[analysis 0002](0002_functional_Faster_R-CNN.md) on the [plate image
dataset](../raw/positives/) and evaluate it on the [Pennfudan
dataset](../raw/pos_control/). The expectation is that it performs
poorly, given pedestrians and plates are sufficiently distinct that a
model trained to detect incubator plates should classify them as
background. This would doubly confirm the model and its associated code
is behaving as expected.

### Goal:

Further validate the functionality of my model and associated code by
implementation of a negative control.

### Hypothesis:

My model - trained on the plate image dataset - should perform very
poorly when evaluated on the Penn Fudan dataset.

### Rationale:

1.  The plate image dataset model has not been trained on the Penn
    Fudan, hence should not be able to classify pedestrians in images.
2.  The two datasets are very different. One contains lambertian objects
    of interest, the other contains transparent, non-lambertian objects
    of interest. Pedestrians and microbe plates are essentially
    completely disparate in their visual presentation.
3.  A poor performance indicates that the model is specific to its
    dataset, indicating it is looking for the plates as it should.

### Experimental Plan:

1.  Load the model trained on the plate image dataset.
2.  Evaluate its performance using COCO metrics.

## Implementation

### Imports

``` python
from plate_detect import helper_training_functions
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
from collections import defaultdict
import torchvision_deps.T_and_utils
```

``` python
root_dir = '/Users/cla24mas/Documents/SC_TSL_15092024_plate_detect'
model, _, _ = helper_training_functions.load_model(root_dir, 2, '0002_02_15_epochs_fix/checkpoint_epoch_9')
model.eval()  # Set model to evaluation mode
```

    FasterRCNN(
      (transform): GeneralizedRCNNTransform(
          Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
          Resize(min_size=(800,), max_size=1333, mode='bilinear')
      )
      (backbone): BackboneWithFPN(
        (body): IntermediateLayerGetter(
          (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
          (layer1): Sequential(
            (0): Bottleneck(
              (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
              (downsample): Sequential(
                (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (1): Bottleneck(
              (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (2): Bottleneck(
              (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
          )
          (layer2): Sequential(
            (0): Bottleneck(
              (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
              (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
              (downsample): Sequential(
                (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
                (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (1): Bottleneck(
              (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (2): Bottleneck(
              (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (3): Bottleneck(
              (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
          )
          (layer3): Sequential(
            (0): Bottleneck(
              (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
              (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
              (downsample): Sequential(
                (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
                (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (1): Bottleneck(
              (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (2): Bottleneck(
              (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (3): Bottleneck(
              (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (4): Bottleneck(
              (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (5): Bottleneck(
              (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
          )
          (layer4): Sequential(
            (0): Bottleneck(
              (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
              (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
              (downsample): Sequential(
                (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
                (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (1): Bottleneck(
              (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (2): Bottleneck(
              (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
          )
        )
        (fpn): FeaturePyramidNetwork(
          (inner_blocks): ModuleList(
            (0): Conv2dNormActivation(
              (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (2): Conv2dNormActivation(
              (0): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (3): Conv2dNormActivation(
              (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (layer_blocks): ModuleList(
            (0-3): 4 x Conv2dNormActivation(
              (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (extra_blocks): LastLevelMaxPool()
        )
      )
      (rpn): RegionProposalNetwork(
        (anchor_generator): AnchorGenerator()
        (head): RPNHead(
          (conv): Sequential(
            (0): Conv2dNormActivation(
              (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): ReLU(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (1): ReLU(inplace=True)
            )
          )
          (cls_logits): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
          (bbox_pred): Conv2d(256, 12, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (roi_heads): RoIHeads(
        (box_roi_pool): MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=(7, 7), sampling_ratio=2)
        (box_head): FastRCNNConvFCHead(
          (0): Conv2dNormActivation(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (1): Conv2dNormActivation(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (2): Conv2dNormActivation(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (3): Conv2dNormActivation(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU(inplace=True)
          )
          (4): Flatten(start_dim=1, end_dim=-1)
          (5): Linear(in_features=12544, out_features=1024, bias=True)
          (6): ReLU(inplace=True)
        )
        (box_predictor): FastRCNNPredictor(
          (cls_score): Linear(in_features=1024, out_features=2, bias=True)
          (bbox_pred): Linear(in_features=1024, out_features=8, bias=True)
        )
      )
    )

Defining the Penn Fudan dataset class as in the [intermediate object
detection PyTorch
tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html):

``` python
class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms): # I like their use of root, this was something I should have done!
        self.root = root
        self.transforms = transforms
        # load all image files
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages")))) # also, note here they sort the otherwise arbitrary os.listdir return - this was a huge flaw I overlooked in my code!
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = read_image(img_path)
        mask = read_image(mask_path)
        # instances are encoded as different colors
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set
        # of binary masks
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        #target["masks"] = tv_tensors.Mask(masks) <--- commented out since my model doen't care about masks
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
```

Taking the evaluation code out of helper_training_functions.train() to
generate evaluation metrics, again evaluating its performance on 50
samples, but fromthe PennFudan dataset instead:

``` python
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model, optimizer, transforms = helper_training_functions.get_model_instance_object_detection(2)

dataset_test = PennFudanDataset('/Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/raw/pos_control/PennFudanPed', transforms)

# split the dataset in train and test set
indices = torch.randperm(len(dataset_test)).tolist()

dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    collate_fn=torchvision_deps.T_and_utils.utils.collate_fn
)

epoch=10
evaluation_across_epochs = defaultdict(list)

eval_metrics = helper_training_functions.evaluate_model(model, data_loader_test, device)
evaluation_across_epochs[f'{epoch}'] = eval_metrics

helper_training_functions.plot_eval_metrics(root_dir, 0, 10, title="Evaluation Metrics at Epoch 10", **{k: v for k, v in evaluation_across_epochs.items()}) # remember models are 0-indexed, so checkpoint_epoch_9 corresponds to the 10th epoch of training

for i in range(0, len(dataset_test)):
    helper_training_functions.plot_prediction(model, dataset_test, device, i, root_dir, 'checkpoint_epoch_9')
```

    creating index...
    index created!
    Test:  [ 0/50]  eta: 0:04:44  model_time: 5.6694 (5.6694)  evaluator_time: 0.0012 (0.0012)  time: 5.6804  data: 0.0097
    Test:  [49/50]  eta: 0:00:05  model_time: 5.8190 (5.8835)  evaluator_time: 0.0012 (0.0017)  time: 5.9197  data: 0.0150
    Test: Total time: 0:04:55 (5.9013 s / it)
    Averaged stats: model_time: 5.8190 (5.8835)  evaluator_time: 0.0012 (0.0017)
    Accumulating evaluation results...
    DONE (t=0.01s).
    IoU metric: bbox
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.001
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.001
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.007
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.007
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.008
    <torchvision_deps.coco_eval.CocoEvaluator object at 0x310e31700>

![](0004_negative_control_files/figure-commonmark/cell-5-output-2.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_0.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-4.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_1.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-6.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_2.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-8.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_3.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-10.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_4.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-12.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_5.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-14.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_6.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-16.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_7.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-18.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_8.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-20.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_9.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-22.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_10.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-24.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_11.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-26.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_12.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-28.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_13.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-30.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_14.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-32.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_15.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-34.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_16.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-36.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_17.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-38.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_18.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-40.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_19.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-42.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_20.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-44.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_21.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-46.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_22.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-48.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_23.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-50.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_24.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-52.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_25.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-54.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_26.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-56.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_27.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-58.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_28.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-60.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_29.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-62.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_30.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-64.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_31.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-66.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_32.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-68.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_33.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-70.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_34.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-72.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_35.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-74.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_36.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-76.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_37.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-78.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_38.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-80.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_39.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-82.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_40.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-84.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_41.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-86.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_42.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-88.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_43.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-90.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_44.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-92.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_45.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-94.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_46.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-96.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_47.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-98.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_48.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-100.png)

    Saved normalized prediction as: /Users/cla24mas/Documents/SC_TSL_15092024_plate_detect/results/prediction_normalized_49.png

![](0004_negative_control_files/figure-commonmark/cell-5-output-102.png)

As you can see in both the stdout from CocoEvaluator, as well as the
plot itself, the model performed extremely poorly as expected (almost
completely imprecise and unable to find the ground-truth bounding boxes
at IoU thresholds \>=0.50):

IoU metric: bbox Average Precision (AP) @\[ IoU=0.50:0.95 \| area= all
\| maxDets=100 \] = 0.001 Average Precision (AP) @\[ IoU=0.50 \| area=
all \| maxDets=100 \] = 0.004 Average Precision (AP) @\[ IoU=0.75 \|
area= all \| maxDets=100 \] = 0.001 Average Precision (AP) @\[
IoU=0.50:0.95 \| area= small \| maxDets=100 \] = -1.000 Average
Precision (AP) @\[ IoU=0.50:0.95 \| area=medium \| maxDets=100 \] =
0.003 Average Precision (AP) @\[ IoU=0.50:0.95 \| area= large \|
maxDets=100 \] = 0.001 Average Recall (AR) @\[ IoU=0.50:0.95 \| area=
all \| maxDets= 1 \] = 0.003 Average Recall (AR) @\[ IoU=0.50:0.95 \|
area= all \| maxDets= 10 \] = 0.016 Average Recall (AR) @\[
IoU=0.50:0.95 \| area= all \| maxDets=100 \] = 0.016 Average Recall (AR)
@\[ IoU=0.50:0.95 \| area= small \| maxDets=100 \] = -1.000 Average
Recall (AR) @\[ IoU=0.50:0.95 \| area=medium \| maxDets=100 \] = 0.018
Average Recall (AR) @\[ IoU=0.50:0.95 \| area= large \| maxDets=100 \] =
0.016
