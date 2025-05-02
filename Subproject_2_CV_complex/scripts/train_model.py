import torch
import torch.nn as nn
import yaml
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd

from model_factory import ModelFactory
import argparse



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

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=10, device="cpu", filename = "test", save_all=True):

    train_losses = []
    test_losses = []
    
    for epoch in tqdm(range(epochs), desc=f"Epoch", position=0):
        # Training phase
        model.train()
        running_train_loss = 0.0

        for images, labels in tqdm(train_loader, desc="Training on images and labels", position=1, leave=False):
            # Move data to GPU
            images = images.float().to(device)
            labels = labels.float().to(device).unsqueeze(1)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
                             
            optimizer.step()
            
            running_train_loss += loss.item()
             
        
        # Calculate average training loss per epoch
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        running_test_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Calculating the test loss", leave=False):
                inputs, labels = inputs.float().to(device), labels.float().to(device).unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_test_loss += loss.item()
        
        # Calculate average test loss per epoch
        epoch_test_loss = running_test_loss / len(test_loader.dataset)
        test_losses.append(epoch_test_loss)

        if save_all:
            torch.save(model.state_dict(), f'./models/files/{filename}-epoch_{epoch}.pth')

    # We need to save the model if we don"t automatically save it at each epoch
    if (save_all==False):
        torch.save(model.state_dict(), f'./models/files/{filename}-epoch_{epoch}.pth')

    return train_losses, test_losses


def model_training(args):
    """General function to run the model training

    Parameters
    ----------
    args : _type_
        Datastructure which holds all the arguments passed to the CLI scipt
    """
    # Initialize the factory (automatically discovers models)
    factory = ModelFactory(models_dir='models')

    # List available models
    print("Available models:", factory.list_models(), end='\n\n')

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", end='\n\n')

    # Extracting the arguments from the CLI
    model_name = args.model_name
    epochs = args.epochs
    plots = args.plots
    filename = args.output_name
    batch_size = args.batch_size
    dataset = args.dataset
    save_all = args.save_all

    # Example 1: Create ResNet
    model_to_train = factory.get_model(model_name).to(device)

    print("Model created:", model_to_train, end='\n\n')

    # Get the name of the files to load for training and testing dataset
    train, test = get_data(dataset)

    # Create the torch dataset
    train_dataset = create_torch_dataset(train)
    test_dataset = create_torch_dataset(test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Define loss function and optimizer
    #criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model_to_train.parameters(), lr=0.001)

    # Train and evaluate
    train_losses, test_losses = train_model(model_to_train, 
                                            train_loader, 
                                            test_loader, 
                                            criterion, 
                                            optimizer, 
                                            epochs=epochs, 
                                            device=device, 
                                            filename = filename,
                                            save_all=save_all)

    if plots:
        loss_df = pd.DataFrame(data=[train_losses, test_losses], columns = [i+1 for i in range(epochs)], index=['Train loss','Test loss'])
        loss_df.T.plot()

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model training with arguments')
    parser.add_argument('--model_name', type=str, default='BinaryIDSModel', help='Model to train')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--plots', type=bool, default=False, help='Do you want to plot the epoch losses')
    parser.add_argument('--output_name', type=str, default="model_state", help='Name of the file to be saved')
    parser.add_argument('--dataset', type=str, default="default", help='Name of the dataset to load')
    parser.add_argument('--save_all', type=bool, default=True, help='Do you want to save models for each epoch')
    
    
    # Parse the arguments passed after -m scripts.test
    args = parser.parse_args()

    model_training(args)