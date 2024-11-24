"""
Plate Detection Model Training Script

This script handles the training of a computer vision model for detecting plates in images.
It sets up data loading, model initialization, and executes the training loop.

The script expects:
- A CSV file containing image annotations (e.g., bounding box coordinates for each sample index)
- A directory of positive sample images
- CUDA-capable GPU (will fall back to CPU if unavailable)

Author: Original by Samuel Clucas, Documentation added by next maintainer
Last Modified: November 2024
"""

from pathlib import Path
from typing import Tuple, List
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.ops.boxes import masks_to_boxes
import numpy as np
import pandas as pd

from plate_detect import (
    Plate_Image_Dataset,
    helper_training_functions
)
import torchvision_deps
from torchvision_deps.T_and_utils import utils


def setup_device() -> torch.device:
    """Configure computation device (GPU if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_paths() -> Tuple[Path, Path, Path]:
    """
    Initialize project directories and files.
    
    Returns:
        Tuple containing:
        - project_root: Base directory of the project
        - annotations_file: Path to CSV containing image labels
        - img_dir: Directory containing training images
    """
    project_root = Path.cwd()
    annotations_file = project_root.joinpath('lib', 'labels.csv')
    img_dir = project_root.joinpath('raw', 'positives')
    
    print(f'Project root directory: {project_root}')
    print(f'Training labels csv file: {annotations_file}')
    print(f'Training dataset directory: {img_dir}')
    
    return project_root, annotations_file, img_dir


def create_datasets(
    dataset: Dataset,
    validation_size: int
) -> Tuple[Subset, Subset]:
    """
    Split dataset into training and validation sets.
    
    Args:
        dataset: Complete dataset to split
        validation_size: Number of samples for validation
    
    Returns:
        Tuple of (training_dataset, validation_dataset)
    """
    dataset_size = len(dataset)
    indices = [int(i) for i in torch.randperm(dataset_size).tolist()]
    
    return (
        Subset(dataset, indices[:-validation_size]),
        Subset(dataset, indices[-validation_size:])
    )


def create_dataloaders(
    train_dataset: Subset,
    val_dataset: Subset,
    batch_size: int = 1
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoader instances for training and validation.
    
    Args:
        train_dataset: Training dataset subset
        val_dataset: Validation dataset subset
        batch_size: Batch size for training (default: 1)
    
    Returns:
        Tuple of (train_loader, validation_loader)
    """
    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=utils.collate_fn
        ),
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=utils.collate_fn
        )
    )


def main():
    """Main execution function for training pipeline."""
    # Setup computation device
    device = setup_device()
    
    # Initialize project paths
    project_root, annotations_file, img_dir = setup_paths()
    
    # Configure data transforms
    transform = transforms.v2.Compose([
        transforms.v2.ToDtype(torch.float32, scale=True)
    ])
    
    # Initialize dataset
    dataset = Plate_Image_Dataset.Plate_Image_Dataset(
        annotations_file=str(annotations_file),
        img_dir=str(img_dir),
        transforms=transform
    )
    
    # Split dataset and create data loaders
    train_dataset, val_dataset = create_datasets(
        dataset=dataset,
        validation_size=50
    )
    
    train_loader, val_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )
    
    # Initialize model (2 classes: background and plate)
    model, optimizer, preprocess = helper_training_functions.get_model_instance_object_detection(
        num_class=2
    )
    model.to(device)
    
    # Execute training loop
    num_epochs = 15
    precedent_epoch = 0
    save_dir = '../'
    
    final_epoch = helper_training_functions.train(
        model=model,
        train_loader=train_loader,
        validation_loader=val_loader,
        device=device,
        num_epochs=num_epochs,
        precedent_epoch=precedent_epoch,
        save_dir=save_dir,
        optimizer=optimizer
    )


if __name__ == '__main__':
    main()