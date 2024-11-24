"""
Helper functions for training and evaluating object detection models.

This module provides utilities for setting up, training, and evaluating
Faster R-CNN models for plate detection. It includes functions for model
initialization, checkpoint management, visualization, and evaluation.

Author: Original by Samuel Clucas, Documentation enhanced
Last Modified: November 2024
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
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.models.detection.faster_rcnn import (
    fasterrcnn_resnet50_fpn_v2,
    FastRCNNPredictor,
    FasterRCNN_ResNet50_FPN_V2_Weights
)
import PIL
from torchvision.utils import draw_bounding_boxes
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms import v2 as T
from collections import defaultdict
from  torchvision_deps.engine import train_one_epoch, evaluate # used by evaluate_model()
from torchvision_deps.T_and_utils import utils
from typing import Tuple, Dict, Any, Optional, List, Type
from torch.utils.data import DataLoader


def get_model_instance_object_detection(
    num_class: int
) -> Tuple[nn.Module, torch.optim.Optimizer, T.Compose]:
    """
    Initialize a Faster R-CNN model with custom classification head.
    
    Args:
        num_class: Number of classes to detect (including background)
        
    Returns:
        Tuple containing:
        - Initialized model
        - SGD optimizer
        - Preprocessing transforms
        
    Note:
        Uses ResNet50 FPN V2 backbone with pretrained weights
    """
    # Initialize model with pretrained weights
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT 
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    
    # Define custom preprocessing transforms
    transforms = T.Compose([
        T.ConvertImageDtype(torch.float32),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Replace classification head for custom number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_class)

    # Configure optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    return model, optimizer, transforms

def train(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 15,
    precedent_epoch: int = 0,
    root_dir: str = '../',
    optimizer: Optional[torch.optim.Optimizer] = None,
    **kwargs: Any
) -> int:
    """
    Train an object detection model.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        validation_loader: DataLoader for validation data
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
        
    This function handles the complete training loop including:
    - Training for specified number of epochs
    - Logging and plotting loss metrics
    - Saving model checkpoints
    - Evaluating model performance
    - Learning rate scheduling
    """
    model.train()
    
    # Initialize metric storage
    losses_across_epochs = defaultdict(list)
    evaluation_across_epochs = defaultdict(list)
    
    # Setup learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=kwargs.get('lr_step_size', 3),
        gamma=kwargs.get('lr_gamma', 0.1)
    )

    for epoch in range(num_epochs):
        # Train one epoch
        metric_logger = train_one_epoch(
            model, 
            optimizer, 
            train_loader,
            device, 
            epoch, 
            print_freq=kwargs.get('print_freq', 1)
        )
        
        # Log metrics
        for k, v in metric_logger.intra_epoch_loss.items():
            print(f"\n Key: {k}\n Value: {v}")
        
        # Store average metrics
        for k, v in metric_logger.meters.items():
            losses_across_epochs[k].append(v.value)
        losses_across_epochs['progression'].append(((epoch+1)/num_epochs)*100)

        # Plot training progress
        plot_training_loss(
            root_dir, 
            epoch, 
            **{k: v for k, v in metric_logger.intra_epoch_loss.items()}
        )

        # Save checkpoint
        save_checkpoint(model, optimizer, epoch + precedent_epoch, root_dir)

        # Evaluate model
        eval_metrics = evaluate_model(model, validation_loader, device)
        evaluation_across_epochs[f'{epoch}'] = eval_metrics

        # Update learning rate
        lr_scheduler.step()

    # Print final evaluation
    {print(k, '\n', v) for k, v in evaluation_across_epochs.items()}

    # Plot final metrics
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

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    root_dir: str
) -> None:
    """
    Save model and optimizer state to a checkpoint file.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch number
        root_dir: Directory to save checkpoint
        
    Creates a 'checkpoints' directory if it doesn't exist and saves the
    checkpoint with epoch number in filename.
    """
    save_path = os.path.join(root_dir, 'checkpoints')
    os.makedirs(save_path, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    filename = f'checkpoint_epoch_{epoch}.pth'
    save_path = os.path.join(save_path, filename)
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


def load_model(
    root_dir: str,
    num_classes: int,
    model_file_name: str
) -> Tuple[nn.Module, torch.optim.Optimizer, int]:
    """
    Load a model from a checkpoint file.
    
    Args:
        root_dir: Directory containing checkpoints
        num_classes: Number of classes in model
        model_file_name: Name of checkpoint file (without .pth extension)
        
    Returns:
        Tuple containing:
        - Loaded model
        - Loaded optimizer
        - Epoch number from checkpoint
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
    """
    model, optimizer, _ = get_model_instance_object_detection(num_classes)
    checkpoint_path = f'{root_dir}/checkpoints/{model_file_name}.pth'
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    return model, optimizer, epoch


def evaluate_model(
    model: nn.Module,
    validation_loader: DataLoader,
    device: torch.device
) -> np.ndarray:
    
    """
    Evaluate model performance using COCO metrics.
    
    Args:
        model: Model to evaluate
        validation_loader: DataLoader for validation data
        device: Device to run evaluation on
        
    Returns:
        numpy array containing COCO evaluation metrics
    """
    model.eval()
    coco_evaluator = evaluate(model, validation_loader, device=device)
    return coco_evaluator.coco_eval['bbox'].stats


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

"""
Visualization functions for training progress and model predictions.

This module provides functions for plotting training losses, evaluation metrics,
feature maps, and model predictions on images.
"""

def plot_training_loss(root_dir: str, epoch: int, **kwargs: Dict[str, List[float]]) -> None:
    """
    Plot training losses during a single epoch.
    
    Args:
        root_dir: Directory to save plot
        epoch: Current epoch number
        **kwargs: Dictionary containing loss metrics including:
            - progression: Percentage through epoch
            - loss_objectness: Objectness loss values
            - loss: Total loss values
            - loss_rpn_box_reg: RPN box regression loss
            - loss_classifier: Classification loss
            - lr: Learning rates
    """
    # Extract metrics from kwargs
    progression = kwargs['progression']
    loss_objectness = kwargs['loss_objectness']
    loss = kwargs['loss']
    loss_rpn_box_reg = kwargs['loss_rpn_box_reg']
    loss_classifier = kwargs['loss_classifier']
    
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


def plot_prediction(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    index: int,
    root_dir: str,
    model_name: str
) -> np.ndarray:
    """
    Plot model predictions on a single image.
    
    Args:
        model: Model to use for predictions
        dataset: Dataset containing images
        device: Device to run model on
        index: Index of image to predict on
        root_dir: Directory to save plot
        model_name: Name of model for plot title
        
    Returns:
        Numpy array of output image with drawn predictions
    """
    # Get image and make prediction
    img, target = dataset[index]
    with torch.no_grad():
        image = img[:3, ...].to(device)
        predictions = model([image])
        pred = predictions[0]
    
    # Normalize image for visualization
    image = image.cpu()
    normalized_image = torch.zeros_like(image)
    for c in range(3):
        channel = image[c]
        min_val = channel.min()
        max_val = channel.max()
        if max_val > min_val:
            normalized_image[c] = (channel - min_val) / (max_val - min_val)
    
    # Convert to uint8 and draw boxes
    image_uint8 = (normalized_image * 255).byte()
    pred_boxes = pred["boxes"].cpu().long()
    output_image = torchvision.utils.draw_bounding_boxes(
        image_uint8,
        pred_boxes,
        colors="red",
        width=3
    )
    
    # Display and save
    output_image = output_image.permute(1, 2, 0).numpy()
    plt.figure(figsize=(15, 10))
    plt.imshow(output_image)
    plt.axis('off')
    plt.savefig(
        f'{root_dir}/results/prediction_normalized_{index}.png',
        bbox_inches='tight',
        pad_inches=0,
        dpi=300
    )
    plt.show()
    plt.close()
    
    return output_image

def plot_eval_metrics(
    root_dir: str,
    precedent_epoch: int,
    num_epochs: int,
    title: Optional[str] = None,
    **kwargs: Dict[str, np.ndarray]
) -> None:
    """
    Plot evaluation metrics across training epochs.
    
    Args:
        root_dir: Directory to save plot
        precedent_epoch: Starting epoch number
        num_epochs: Total number of epochs
        title: Optional custom plot title
        **kwargs: Dictionary containing evaluation metrics for each epoch
            Each value should be a numpy array containing COCO metrics
    
    Note:
        Plots AP (Average Precision) and AR (Average Recall) metrics
        from COCO evaluation results
    """
    progression = [k for k in kwargs]  # Epoch numbers
    
    # Extract metrics of interest
    AP = [v[0] for k, v in kwargs.items()]  # AP @ IoU 0.50:0.95
    AR = [v[8] for k, v in kwargs.items()]  # AR @ IoU 0.50:0.95
    
    plt.figure(figsize=(10, 6))
    plt.plot(progression, AP, label='AP @ IoU 0.50:0.95, area=all, maxDets=100')
    plt.plot(progression, AR, label='AR @ IoU 0.50:0.95, area=all, maxDets=100')

    plt.xlabel('Epochs')
    plt.ylabel('Value')
    
    plot_title = (title if title is not None 
                 else f'Evaluation Metrics from Epoch {precedent_epoch}-{precedent_epoch + num_epochs}')
    plt.title(plot_title)
    plt.legend()
    plt.grid(False)
    plt.savefig(f'{root_dir}/results/evaluation_metrics_epochs_{precedent_epoch}-{precedent_epoch + num_epochs}')
    plt.show()


def plot_losses_across_epochs(
    root_dir: str,
    precedent_epoch: int,
    epochs: int,
    **kwargs: Dict[str, List[float]]
) -> None:
    """
    Plot training losses across multiple epochs.
    
    Args:
        root_dir: Directory to save plot
        precedent_epoch: Starting epoch number
        epochs: Total number of epochs
        **kwargs: Dictionary containing loss metrics including:
            - progression: Progress through training
            - loss_objectness: Objectness loss values
            - loss: Total loss values
            - loss_rpn_box_reg: RPN box regression loss
            - loss_classifier: Classification loss
            - lr: Learning rates
    """
    # Extract metrics
    progression = kwargs['progression']
    loss_objectness = kwargs['loss_objectness']
    loss = kwargs['loss']
    loss_rpn_box_reg = kwargs['loss_rpn_box_reg']
    loss_classifier = kwargs['loss_classifier']
    
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

def get_feature_maps(
    model: nn.Module,
    input_image: torch.Tensor,
    target_layer_name: Type[nn.Module]
) -> Dict[nn.Module, torch.Tensor]:
    """
    Extract feature maps from specific layers of the model.
    
    Args:
        model: Model to extract features from
        input_image: Input image tensor
        target_layer_name: Type of layer to extract features from
            (e.g., nn.Conv2d, nn.BatchNorm2d, nn.ReLU)
            
    Returns:
        Dictionary mapping layer instances to their feature maps
    
    Note:
        Uses forward hooks to capture intermediate layer outputs
        during model forward pass
    """
    feature_maps = {}
    
    def hook_fn(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        """Store the output of the layer in feature_maps."""
        feature_maps[module] = output.detach()
    
    # Register hooks for target layers
    hooks = []
    for name, module in model.backbone.named_modules():
        if isinstance(module, target_layer_name):
            hooks.append(module.register_forward_hook(hook_fn))
    
    # Forward pass
    with torch.no_grad():
        model([input_image])
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return feature_maps


def visualise_feature_maps(
    feature_maps: Dict[nn.Module, torch.Tensor],
    num_features: int = 64
) -> None:
    """
    Visualize feature maps from model layers.
    
    Args:
        feature_maps: Dictionary of feature maps from model layers
        num_features: Maximum number of features to display (default: 64)
        
    Creates an 8x8 grid of feature maps for each layer, displaying
    up to num_features maps.
    """
    for layer, feature_map in feature_maps.items():
        # Get first image in batch
        feature_map = feature_map[0]
        
        # Limit number of features to display
        num_to_plot = min(feature_map.size(0), num_features)
        
        fig, axs = plt.subplots(8, 8, figsize=(20, 20))
        fig.suptitle(f'Feature Maps for Layer: {layer}')
        
        for i in range(num_to_plot):
            ax = axs[i // 8, i % 8]
            ax.imshow(feature_map[i].cpu(), cmap='gray')
            ax.axis('off')
        
        # Clear unused subplots
        for i in range(num_to_plot, 64):
            axs[i // 8, i % 8].axis('off')
        
        plt.tight_layout()
        plt.show()