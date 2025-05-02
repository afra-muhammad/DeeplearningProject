import yaml
from torch.utils.data import TensorDataset
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def reshape_xXy(series, x=12, y=12):
    return pd.DataFrame(series.values.reshape(x,y))

def load_config():
    """Function to load the configuration files for the models

    Returns
    -------
    Dict
        Dictionnary that hold the configuration
    """
    # Get the path to the config file (assuming it's in the same directory)
    config_path =  "dataset_config.yaml"
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


def get_data(dataset:str):
    """Extracting the name of the data files for a dataset

    Parameters
    ----------
    dataset : str
        dataset name

    Returns
    -------
    str,str
        Path names for the train and test data

    Raises
    ------
    ValueError
        If the key is not implemented, it will raise a ValueError
    """
    config = load_config()
    try:
        return config['datasets'][dataset]["train_dataset"], config['datasets'][dataset]["test_dataset"]
    except KeyError:
        raise ValueError(f"Data file key '{dataset}' or environment '{dataset}' not found in config")

def create_torch_dataset(path:str):
    """Create a torch dataset

    Parameters
    ----------
    path : str
        Path of the raw dataset

    Returns
    -------
    TensorDataset
        Pytorch TensorDataset which will be used in model training
    """
    print(f'Reading data from {path}')

    # Read the parquet file
    data = pd.read_parquet(path)
    
    # Extract the data (wihtout labels) and reshape them in 11x11 images
    images = np.stack(data.drop(['Label'], axis=1).values).reshape(-1, 1,11,11)
    # Extract the labels
    labels = data.loc[:,'Label'].values

    # Create the dataset
    dataset = TensorDataset(   torch.from_numpy(images), 
                                torch.from_numpy(labels))

    return dataset


def grad_cam(model, img, target_class=None, device="cpu"):
    img = img.requires_grad_(True)  # Add batch dim and enable grad

    # Forward pass
    output = model(img.float().to(device))

    if target_class is None:
        target_class = torch.argmax(output)
    score = output[0, target_class]

    # Backward pass to get gradients
    model.zero_grad()
    score.backward()
    
    # Get the last convolutional layer (modify as per your model)
    last_conv_layer = None
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            last_conv_layer = layer

    # Gradients and feature maps
    gradients = img.grad  # Gradients w.r.t input

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])  # Global avg pooling
    
    # Weight the feature maps
    feature_maps = last_conv_layer(img.float().to(device))
    #feature_maps = last_conv_layer(img.to(device).float())

    for i in range(feature_maps.shape[1]):
        feature_maps[:, i, :, :] *= pooled_gradients[i]
    
    # Generate heatmap
    heatmap = torch.mean(feature_maps, dim=1).squeeze()
    heatmap = F.relu(heatmap)  # Apply ReLU
    heatmap /= torch.max(heatmap)  # Normalize
    
    return heatmap.cpu().detach().numpy()

def create_heatmap(x, model, device="cpu"):
    images = torch.stack([x[0]])

    heatmap = grad_cam1(model, images, device=device, guided=True)

    return heatmap

def extract_sub_dataset(tensordataset, mask):  
    """Extract a subset of the data based on a mask

    Parameters
    ----------
    tensordataset : TEnsorDataset
        Input dataset
    mask : 
        Mask to extract data

    Returns
    TensorDataset
        subset of data
    """
    from torch.utils.data import TensorDataset

    # Apply the mask to both x and y tensors
    filtered_x = tensordataset.tensors[0][mask]
    filtered_y = tensordataset.tensors[1][mask]

    # Create a new filtered dataset
    filtered_dataset = TensorDataset(filtered_x, filtered_y)

    return filtered_dataset


def grad_cam1(model, img, target_class=None, device="cpu", guided=False):
    """Generate Grad-CAM or Guided Grad-CAM heatmap.
    
    Args:
        model: PyTorch model.
        img: Input tensor (batch dim included).
        target_class: Class index to explain (None = predicted class).
        device: 'cpu' or 'cuda'.
        guided: If True, returns Guided Grad-CAM (higher resolution).
    
    Returns:
        heatmap (numpy array): Normalized heatmap (5×5 for Grad-CAM, 11×11 for Guided).
    """
    model.eval()
    
    # Forward pass (enable gradient tracking for input)
    img = img.float().to(device).requires_grad_(True)
    output = model(img)
    
    if target_class is None:
        target_class = torch.argmax(output)
    score = output[0, target_class]

    # --- Grad-CAM ---
    # Find last conv layer
    last_conv = None
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            last_conv = layer
    
    # Hook to capture feature maps and gradients
    feature_maps = []
    gradients = []
    
    def forward_hook(module, inp, out):
        feature_maps.append(out.detach())
    
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())
    
    # Register hooks
    forward_handle = last_conv.register_forward_hook(forward_hook)
    backward_handle = last_conv.register_full_backward_hook(backward_hook)
    
    # Forward + backward pass
    model.zero_grad()
    output = model(img)
    score = output[0, target_class]
    score.backward(retain_graph=guided)
    
    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()
    
    # Check if we got data
    if not feature_maps or not gradients:
        raise ValueError("No feature maps or gradients captured")
    
    # Get gradients and feature maps
    grads = gradients[0]
    feats = feature_maps[0]
    
    # Pool gradients spatially (average over H,W)
    pooled_grads = torch.mean(grads, dim=[0, 2, 3], keepdim=True)
    
    # Ensure dimensions match
    if pooled_grads.shape[1] != feats.shape[1]:
        # Handle channel mismatch (for BinaryIDSModel)
        pooled_grads = pooled_grads.mean(dim=1, keepdim=True)  # Average across channels
        pooled_grads = pooled_grads.expand_as(feats)  # Match feature dimensions
    
    # Weight features by importance
    weighted_feats = feats * pooled_grads
    heatmap = torch.mean(weighted_feats, dim=1, keepdim=True)
    heatmap = F.relu(heatmap).squeeze()
    
    # Normalize
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    if not guided:
        return heatmap.cpu().detach().numpy()
    
    # --- Guided Grad-CAM ---
    input_grads = img.grad.abs().mean(dim=1, keepdim=True)  # [1, 1, H, W]
    
    # Upsample heatmap to input size
    heatmap_upsampled = F.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0),
        size=img.shape[2:],
        mode='bilinear',
        align_corners=False
    ).squeeze()
    
    guided_gradcam = heatmap_upsampled * input_grads.squeeze()
    guided_gradcam = (guided_gradcam - guided_gradcam.min()) / (guided_gradcam.max() - guided_gradcam.min() + 1e-8)
    
    return guided_gradcam.cpu().detach().numpy()