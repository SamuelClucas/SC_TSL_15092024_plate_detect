import os
from lib.engine import train_one_epoch, evaluate 
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
import lib.utils as utils
from lib.engine import train_one_epoch, evaluate 
import PIL
from torchvision.utils import draw_bounding_boxes
import torchvision
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib

def get_model_instance_bounding_boxes(num_class: int) -> fasterrcnn_resnet50_fpn_v2:
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

def train(dataset, model, num_epochs, precedent_epoch, device, data_loader_test, data_loader):
    save_dir = './checkpoints/' 
    model.train()
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
        running_loss = 0.0
        for i, data in enumerate(data_loader_test):
            running_loss += loss.item() 
    # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

        save_checkpoint(model, optimizer, epoch, save_dir)
    # update the learning rate
        lr_scheduler.step()
    # evaluate on the test dataset
        model.eval()
    evaluate(model,device, data_loader_test)

    epoch += (precedent_epoch-1)

    return optimizer, epoch

def load_model(save_dir):
    model, preprocess = get_model_instance_bounding_boxes(2)
    checkpoint = torch.load(save_dir + '/checkpoint_epoch_1.pth', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch']
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

def plot_prediction(model, dataset, device):
    img, target = dataset[6]
    num_epochs = 1
    print_freq = 10
    save_dir = './checkpoints'  

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

def tensorboard_summary(run_name, dataset, model, data_loader) -> SummaryWriter:  # https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    writer = SummaryWriter(f'runs/{run_name}')

    image, target = dataset[1]

    dataiter = iter(data_loader)
    images, targets = next(dataiter)
    

    writer.add_graph(model, images[0])
    writer.close()



