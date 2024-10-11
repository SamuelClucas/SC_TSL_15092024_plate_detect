# Helper Training Function

8/9/25

### Purpose: 

I created numerous helper functions for import into training/evaluation
scripts. This document serves to explain each one.  

### Programme Overview:

#### Imports:

``` python
from torchvision_deps.engine import train_one_epoch, evaluate 
import re
from glob import glob
import numpy as np
import pandas as pd
from torchvision.io import read_image
import torch
from torchvision.transforms.v2 import functional as F
from torchvision import tv_tensors
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Union
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2, FastRCNNPredictor, FasterRCNN_ResNet50_FPN_V2_Weights
import torchvision_deps.T_and_utils.utils as utils 
import PIL
from torchvision.utils import draw_bounding_boxes
import torchvision
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
```

#### Get Faster R-CNN model instance for object detection…

``` python
def get_model_instance_object_detection(num_class: int) -> fasterrcnn_resnet50_fpn_v2:
    # New weights with accuracy 80.858%
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT # alias is .DEFAULT suffix, weights = None is random initialisation, box MAP 46.7, params, 43.7M, GFLOPS 280.37 https://github.com/pytorch/vision/pull/5763
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.0001)
    preprocess = weights.transforms()
    # finetuning pretrained model by transfer learning
    # get num of input features for classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_class)
    return model, preprocess
```

**Breakdown:**  
- this approach requires permissions to download the model file from the
internet (through ‘torch.utils.model_zoo.load_url()’). For this reason,
it is not amenable to training on the cluster. This will be addressed in
analysis 0003, using a local Faster R-CNN with ResNet backbone class
definition. Otherwise, this function still works.  
- See pytorch’s documentation on [ResNet50 faster R-CNN
backbone](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html).

#### Save function

``` python
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
```

**Breakdown:**  
- Creates a directory if it doesn’t exist to store the checkpoints.  
- Saves the current epoch number, model parameters (weights), and
optimizer state to a dictionary.  
- This allows for resuming training from a specific epoch in case of
interruptions.  

#### Training function:

``` python
def train(model, data_loader, data_loader_test, device, num_epochs, precedent_epoch, save_dir): 
    model.train()

    # Initialize lists to store metrics
    train_losses = []

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

    # train for one epoch, printing every 10 iterations
    metric_logger = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    
    # Get the average loss for this epoch
    epoch_loss = metric_logger.meters['loss'].global_avg
    train_losses.append(epoch_loss)

    # Save checkpoint
    save_checkpoint(model, optimizer, epoch + precedent_epoch, save_dir)
        
    # Plot and save the metrics
    plot_eval_metrics(train_losses, epoch + precedent_epoch)
        
    # Update the learning rate
    lr_scheduler.step()

    return epoch + precedent_epoch, train_losses
```

**Breakdown:**  
Setting the Model to Training Mode:  
- model.train() sets the model to training mode. This is crucial because
certain layers (like dropout and batch normalization) behave differently
during training versus evaluation.  

Optimizer Construction:  
- The optimizer is created using Stochastic Gradient Descent (SGD) with
a learning rate of 0.005, momentum of 0.9, and weight decay for
regularization (also known as [L2
regularisation](https://arxiv.org/pdf/2310.04415)).  
- params filters the model’s parameters to include only those that
require gradients (trainable parameters).  

Learning Rate Scheduler:  
- A learning rate scheduler (StepLR) is initialized, which reduces the
learning rate by a factor of gamma (0.1) every step_size epochs (3
epochs in this case). This helps improve convergence as training
progresses.  

Epoch Loop:  
- The outer loop iterates over the number of specified epochs
(num_epochs).  

Training for One Epoch:  
- train_one_epoch(…) is a function from
[engine.py](../../src/torchvision_deps/engine.py) that handles the
training logic for one epoch. It processes batches from the data_loader,
computes losses, and updates model weights.  
- The print_freq parameter controls how often training progress is
printed (every 10 iterations).  

Logging Loss:  
- The average loss for the epoch is extracted from metric_logger, which
tracks various metrics during training.  

Checkpoint Saving:  
- After each epoch, a checkpoint is saved using the save_checkpoint
function. This allows for resuming from the last epoch in case of
interruption.  

Metrics Plotting:  
- The function plot_eval_metrics is called to visualize training losses
over epochs.  

Learning Rate Update:  
- After the epoch, the learning rate is updated according to the
scheduler.  

Return Values:  
- The function returns the final epoch number (adjusted for any
precedent epochs) and the list of training losses.  

#### Load a model from a .pth file:

``` python
def load_model(save_dir: str, num_classes: int, model_file_name: str):
    model, preprocess = get_model_instance_bounding_boxes(num_classes)
    checkpoint = torch.load(save_dir + f'{model_file_name}.pth', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch
```

**Breakdown:**  
- Loads a model from a .pth file. See the pytorch documentation
[here](https://pytorch.org/tutorials/beginner/saving_loading_models.html).  
- The weights and biases of a model are accessed by model.parameters().
A state_dict is a dictionary object that maps each layer to its
parameter tensor. Both the model and optimiser have state_dicts, and
both are saved by the save_checkpoint helper function.  
- Returns the model, optimizer and epoch.  

#### Evaluate a model’s performance after training:

``` python
def evaluate_model(model, data_loader_test: torch.utils.data.DataLoader, device: torch.cuda.device): 
    val_metrics = []
    model.eval()

    coco_evaluator = evaluate(model, data_loader_test, device=device)
        
    # Extract evaluation metrics
    eval_stats = coco_evaluator.coco_eval['bbox'].stats
    val_metrics.append(eval_stats)
    return val_metrics
```

**Breakdown:**  
- Sets the model to evaluation mode using model.eval(), which alters the
behavior of certain layers.  
- In training mode, the model expects input tensors (images) and targets
(list of dictionary containing bounding boxes \[N, 4\], with vertices in
the format \[x1, y1, x2, y2\] and class labels in the format
Int64Tensor\[N\] where N is the number of bounding boxes in a given
image, or the number of distinct classes, respectively).  
- Currently using [coco_eval](../../src/torchvision_deps/).  
- See device class docs
[here](https://pytorch.org/docs/stable/generated/torch.cuda.device.html)  
- Makes a call to [engine’s](../../src/torchvision_deps/engine.py)
evaluator() to perform COCO-style evaluation on the test dataset, shown
below:  

``` python
@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ") # MetricLogger: https://pytorch.org/tnt/stable/utils/generated/torchtnt.utils.loggers.MetricLogger.html
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
```

- COCO provides thorough documentation
  [here](https://cocodataset.org/#detection-eval).
- coco_evaluator.summarize() computes Average Precision (AP), AP Across
  Scales, Average Recall (AR), AR Across Scales (see a dicussion of
  these metrics
  [here](https://blog.zenggyu.com/posts/en/2018-12-16-an-introduction-to-evaluation-metrics-for-object-detection/index.html)).

#### Using PyTorch hooks to get backbone feature maps:

``` python
def get_feature_maps(model, input_image, target_layer_name: torch.nn):
    feature_maps = {} # stores activations passed to forward_hook
    
    def hook_fn(module, input, output):
        feature_maps[module] = output.detach()
    return hook_fn # calls handle.remove() to remove the added hook
    
    # Register hooks for the layers to be visualised
    for name, module in model.backbone.named_modules():
        if isinstance(module, target_layer_name): # type examples: nn.Conv2d, nn.BatchNorm2d, nn.ReLU
            module.register_forward_hook(hook_fn(module))
    
    # Forward pass
    with torch.no_grad(): # disables loss gradient calculation: https://discuss.pytorch.org/t/with-torch-no-grad/130146
        model([input_image])
    
    return feature_maps
```

**Breakdown:**  
- In order to extract intermediate activations from model layers,
PyTorch provides ‘hooks’. [Pytorch documentation on
hooks](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook)
is quite sparse, but
[this](https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/)
blog is quite useful.  
- Gets the feature maps for every target layer (target_layer_name) in
each named module in the model’s backbone (e.g., ‘Sequential’,
‘Bottleneck’). Target layers are documented
[here](https://pytorch.org/docs/stable/nn.html#convolution-layers).  

#### Main:

``` python
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
```

**Breakdown:**  
- This function takes the feature maps collected from the
get_feature_maps function and visualizes them.  
- It plots up to num_features from each layer’s output, providing visual
insights into what features the model is learning at different levels.

#### Superimpose and plot bounding boxes on image:

``` python
def plot_prediction(model, dataset, device, save_dir: str):
    img, target = dataset[6]
    num_epochs = 1
    print_freq = 10  

    model = load_model(save_dir)
    with torch.no_grad():
        image = img
        image = image[:3, ...].to(device)
        predictions = model([image, ])
        pred = predictions[0]

    image = image[:3, ...]
    pred_boxes = pred["boxes"].long()
    output_image = draw_bounding_boxes(image, pred_boxes, colors="red")
    plt.figure(figsize=(12, 12))
    plt.imshow(output_image.permute(1, 2, 0))
    plt.show()
```

**Breakdown:**  
- Loads a specific image from the dataset and runs it through the model
to get predictions.  
- Superimposes the predicted bounding boxes on the original image (as a
red outlined bounding box) and visualises the result.  

#### Main:

``` python
def plot_eval_metrics(eval_metrics, epoch):
    plt.figure(figsize=(15, 10))
    
   # # Plot training loss
   # plt.subplot(2, 2, 1)
   # plt.plot(train_losses, label='Train Loss')
   # plt.title('Training Loss over epochs')
   # plt.xlabel('Epoch')
   # plt.ylabel('Loss')
   # plt.legend()
    
    # Plot mAP@[0.5:0.95]
    plt.subplot(2, 2, 2)
    map_values = [metrics[0] for metrics in eval_metrics]
    plt.plot(map_values, label='mAP@[0.5:0.95]')
    plt.title('mAP@[0.5:0.95] over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()
    
    # Plot mAP@0.5
    plt.subplot(2, 2, 3)
    map50_values = [metrics[1] for metrics in eval_metrics]
    plt.plot(map50_values, label='mAP@0.5')
    plt.title('mAP@0.5 over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()
    
    # Plot mAR@[0.5:0.95]
    plt.subplot(2, 2, 4)
    mar_values = [metrics[8] for metrics in eval_metrics]
    plt.plot(mar_values, label='mAR@[0.5:0.95]')
    plt.title('mAR@[0.5:0.95] over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('mAR')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'training_metrics_epoch_{epoch}.png')
    plt.close()
```

**Breakdown:**  
- This function visualizes the evaluation metrics (mean Average
Precision (mAP) and mean Average Recall (mAR)) over epochs.  
- The results are plotted for easy comparison, helping to analyze the
training process and the model’s learning behavior over time.  

### Next step: [implementing these functions in a setup script…](03_ResNet50_setup.md)