import os
import re
from glob import glob
import numpy as np
import pandas as pd
from torchvision.io import read_image
import torch
from torchvision.transforms.v2 import functional as F
from torchvision import tv_tensors
import matplotlib.pyplot as plt
from typing import Dict
from pathlib import Path
from typing import Dict, Union
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2, FastRCNNPredictor, FasterRCNN_ResNet50_FPN_V2_Weights
import PIL
from torchvision.utils import draw_bounding_boxes
import torchvision
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchvision.transforms import v2 as T
from collections import defaultdict
from  torchvision_deps.engine import train_one_epoch, evaluate # used by evaluate_model()
from torchvision_deps.T_and_utils import utils
from typing import Dict, Any, Optional

def get_model_instance_object_detection(num_class: int):
    # New weights with accuracy 80.858%
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT 
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    
    # Don't use the default preprocess from weights
    # Instead, use our custom transforms we just tested
    transforms = T.Compose([
        T.ConvertImageDtype(torch.float32),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # get num of input features for classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_class)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    return model, optimizer, transforms

def save_checkpoint(model, optimizer, epoch, root_dir):
    save_path = os.path.join(root_dir, 'checkpoints')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    filename = f'checkpoint_epoch_{epoch}.pth'
    save_path = os.path.join(save_path, filename)
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")

def plot_training_loss(root_dir: str, epoch: int, **kwargs):
    progression = [v for v in kwargs['progression']] # I'm not sure if list comprehension is neccessary here... perhaps just try progression = kwargs['progression']?
    loss_objectness = [v for v in kwargs['loss_objectness']]
    loss = [v for v in kwargs['loss']]
    loss_rpn_box_reg = [v for v in kwargs['loss_rpn_box_reg']]
    loss_classifier = [v for v in kwargs['loss_classifier']]
    lr = [v for v in kwargs['lr']] 

    # the metrics are passed using list comprehension in train()
    plt.figure(figsize=(10, 6))
    plt.plot(progression, loss, label='Loss')
    plt.plot(progression, loss_objectness, label='Loss Objectness')
    plt.plot(progression, loss_rpn_box_reg, label='Loss RPN Box Reg')
    plt.plot(progression, loss_classifier, label='Loss Classifier')

    plt.xlabel('Progression through Epoch / %')
    plt.ylabel('Loss / Log10')
    plt.yscale("log")
    plt.title(f'Loss Metrics Progression Through Epoch {epoch}')
    plt.legend()
    plt.grid(False)
    plt.savefig(f'{root_dir}/results/loss_metrics_epoch_{epoch}')
    plt.show()

def plot_losses_across_epochs(root_dir: str, precedent_epoch: int, epochs: int, **kwargs):
    progression = [v for v in kwargs['progression']] # I'm not sure if list comprehension is neccessary here... perhaps just try progression = kwargs['progression']?
    loss_objectness = [v for v in kwargs['loss_objectness']]
    loss = [v for v in kwargs['loss']]
    loss_rpn_box_reg = [v for v in kwargs['loss_rpn_box_reg']]
    loss_classifier = [v for v in kwargs['loss_classifier']]
    lr = [v for v in kwargs['lr']] 
    
    # the metrics are passed using list comprehension in train()
    plt.figure(figsize=(10, 6))
    plt.plot(progression, loss, label='Loss')
    plt.plot(progression, loss_objectness, label='Loss Objectness')
    plt.plot(progression, loss_rpn_box_reg, label='Loss RPN Box Reg')
    plt.plot(progression, loss_classifier, label='Loss Classifier')

    plt.xlabel('Epochs')
    plt.ylabel('Loss / Log10')
    plt.yscale("log")
    plt.title(f'Loss Metrics from Epoch {precedent_epoch}-{precedent_epoch + epochs}')
    plt.legend()
    plt.grid(False)
    plt.savefig(f'{root_dir}/results/loss_metrics_epochs_{precedent_epoch}-{precedent_epoch + epochs}')
    plt.show()

def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader, 
    validation_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_epochs: int = 15,
    precedent_epoch: int = 0,
    root_dir: str = '../',
    optimizer: Optional[torch.optim.Optimizer] = None,
    **kwargs
) -> int:
    """
    Train an object detection model.
    
    Args:
        model: The model to train
        data_loader: Training data loader
        data_loader_test: Validation/test data loader
        device: Device to train on (CPU/GPU)
        num_epochs: Number of epochs to train (default: 15)
        precedent_epoch: Starting epoch number (default: 0)
        root_dir: Directory for saving outputs (default: '../')
        optimizer: Optimizer for training
        **kwargs: Additional arguments including:
            - lr_step_size: Learning rate scheduler step size (default: 3)
            - lr_gamma: Learning rate decay factor (default: 0.1)
            - print_freq: Frequency of printing training progress (default: 1)
    
    Returns:
        int: Final epoch number
    """
    model.train()
    
    # Initialize metric storage
    losses_across_epochs = defaultdict(list)
    evaluation_across_epochs = defaultdict(list)
    
    # Set up learning rate scheduler with configurable parameters
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=kwargs.get('lr_step_size', 3),
        gamma=kwargs.get('lr_gamma', 0.1)
    )

    for epoch in range(num_epochs):
        # Train for one epoch with configurable print frequency
        metric_logger = train_one_epoch(
            model, 
            optimizer, 
            train_loader , 
            device, 
            epoch, 
            print_freq=kwargs.get('print_freq', 1)
        )
        
        # Log intra-epoch metrics
        for k, v in metric_logger.intra_epoch_loss.items():
            print(f"\n Key: {k}\n Value: {v}")
            
        # Store average loss metrics
        for k, v in metric_logger.meters.items():
            losses_across_epochs[k].append(v.value)
        losses_across_epochs['progression'].append(((epoch+1)/num_epochs)*100)

        # Plot training loss
        plot_training_loss(
            root_dir, 
            epoch, 
            **{k: v for k, v in metric_logger.intra_epoch_loss.items()}
        )

        # Save model checkpoint
        save_checkpoint(model, optimizer, epoch + precedent_epoch, root_dir)

        # Evaluate model on test set
        eval_metrics = evaluate_model(model, validation_loader, device)
        evaluation_across_epochs[f'{epoch}'] = eval_metrics

        # Update learning rate
        lr_scheduler.step()

    # Print evaluation metrics
    {print(k, '\n', v) for k, v in evaluation_across_epochs.items()}

    # Plot metrics across epochs
    plot_losses_across_epochs(
        root_dir, 
        precedent_epoch, 
        num_epochs, 
        **{k: v for k, v in losses_across_epochs.items()}
    )
    plot_eval_metrics(
        root_dir, 
        precedent_epoch, 
        num_epochs, 
        **{k: v for k, v in evaluation_across_epochs.items()}
    )
    
    return num_epochs + precedent_epoch


def load_model(root_dir: str, num_classes: int, model_file_name: str):
    model, optimizer, preprocess = get_model_instance_object_detection(num_classes)
    checkpoint = torch.load(f'{root_dir}/checkpoints/{model_file_name}.pth', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

def evaluate_model(model, validation_loader,device):
    model.eval()

    coco_evaluator = evaluate(model, validation_loader, device=device)
    print(coco_evaluator)
    # Extract evaluation metrics
    eval_metrics = coco_evaluator.coco_eval['bbox'].stats
  
    return eval_metrics


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

def visualise_feature_maps(feature_maps, num_features=64):
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

def plot_prediction(model, dataset, device, index, root_dir: str, model_name):
    # Get image and target from dataset
    img, target = dataset[index]
    
    with torch.no_grad():
        image = img[:3, ...].to(device) # take the first 3 elements/channels (RGB) - leave the rest as the same (the '...')
        predictions = model([image])
        pred = predictions[0]
    
    # Move image back to CPU for visualization
    image = image.cpu()
    
    # Normalize each channel independently
    normalized_image = torch.zeros_like(image)
    for c in range(3):
        channel = image[c]
        min_val = channel.min()
        max_val = channel.max()
        if max_val > min_val:
            normalized_image[c] = (channel - min_val) / (max_val - min_val)
    
    # Convert to 8-bit format for visualization
    image_uint8 = (normalized_image * 255).byte()
    
    # Get predicted boxes
    pred_boxes = pred["boxes"].cpu().long()
    
    # Draw boxes on image
    output_image = torchvision.utils.draw_bounding_boxes(
        image_uint8,
        pred_boxes,
        colors="red",
        width=3
    )
    
    # Convert tensor to numpy and ensure proper format for matplotlib
    output_image = output_image.permute(1, 2, 0).numpy()
    
    # Create figure and display
    plt.figure(figsize=(15, 10))
    plt.imshow(output_image)
    plt.axis('off')
    
    # Save the figure
    plt.savefig(f'{root_dir}/results/prediction_normalized_{index}.png', bbox_inches='tight', pad_inches=0, dpi=300)
    print(f"Saved normalized prediction as: {root_dir}/results/prediction_normalized_{index}.png")
    
    plt.show()
    plt.close()

    return output_image

def plot_eval_metrics(root_dir, precedent_epoch, num_epochs, title=None, **kwargs):
    progression = [k for k in kwargs] # should be equivalent to num_epochs - precedent_epoch

    # store metrics of interest in lists
    AP = [v[0] for k, v in kwargs.items()] 
    AR = [v[8] for k, v in kwargs.items()]
    
    plt.figure(figsize=(10, 6))
    plt.plot(progression, AP, label='AP @ IoU 0.50:0.95, area=all, maxDets=100')
    plt.plot(progression, AR, label='AR @ IoU 0.50:0.95, area=all, maxDets=100')

    plt.xlabel('Epochs')
    plt.ylabel('Value')
    
    if title == None:
        plt.title(f'Evaluation Metrics from Epoch {precedent_epoch}-{precedent_epoch + num_epochs}')
    else:
        plt.title(title) # lets you plot just one epoch's training metrics
    plt.legend()
    plt.grid(False)
    plt.savefig(f'{root_dir}/results/evaluation_metrics_epochs_{precedent_epoch}-{precedent_epoch + num_epochs}')
    plt.show()

