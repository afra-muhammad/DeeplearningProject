import argparse
from model_factory import ModelFactory
import torch
from torch.utils.data import DataLoader
from .utils import get_data, create_torch_dataset, create_heatmap, extract_sub_dataset
from tqdm import tqdm
import numpy as np
from .models_utils import load_model, plot_confusion_matrix
from .plot_utils import plot_attention_map

def evaluate_model(args):
    # Extracting the arguments from the CLI
    model_name = args.model_name
    dataset = args.dataset
    batch_size = args.batch_size
    model_file = args.model_file
    random_model = args.random_model
    attention_map = args.attention_map

    # Get the name of the files to load for training and testing dataset
    _, test = get_data(dataset)

    # Create the torch dataset
    test_dataset = create_torch_dataset(test)

    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize the factory (automatically discovers models)
    factory = ModelFactory(models_dir='models')

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", end='\n\n')

    # Load the model to evaluate
    model_to_evaluate = load_model(factory, 
                                   model_name, 
                                   model_file, 
                                   random_model=random_model, 
                                   device=device)

    # Plot the confusion matrix on the test set
    plot_confusion_matrix(model_to_evaluate, test_loader, device=device)


    # We check the attention map only in the case of non random model and we want to perform it
    if (random_model == False) and (attention_map==True):
        print("\n")
        # Extract the benign and malicious dataset from the test data
        labels = test_dataset.tensors[1]  # Assuming y is the second element in each sample

        # Creating a mask for benign data
        mask = (labels == 0)
        benign_dataset = extract_sub_dataset(test_dataset, mask)

        # Creating a mask for malicious attacks
        mask = (labels == 1)
        malicious_dataset = extract_sub_dataset(test_dataset, mask)


        # Loading the model to evaluate
        model_to_evaluate = load_model(factory, model_name, model_file, random_model=False, device=device)


        heatmap_benign = [create_heatmap(x, 
                                            model_to_evaluate, device=device) for x in tqdm(benign_dataset)]
        
        heatmap_malicious = [create_heatmap(x, 
                                            model_to_evaluate, device=device) for x in tqdm(malicious_dataset)]
        
        

        benign_average = np.mean(heatmap_benign, axis=0)
        malicious_average = np.mean(heatmap_malicious, axis=0)
        
        plot_attention_map(benign_average, malicious_average)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model training with arguments')
    parser.add_argument('--model_name', type=str, default='BinaryIDSModel', help='Model to train')
    parser.add_argument('--dataset', type=str, default="default", help='Name of the dataset to load')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--model_file', type=str, default="model_state_dict", help='Name of the model file to load')
    parser.add_argument('--random_model', type=bool, default=False, help='Set it to true to see the result of a non trained model')
    parser.add_argument('--attention_map', type=bool, default=False, help='Set it to true to calculate attention map (long calculations)')
    
    # Parse the arguments passed after -m scripts.test
    args = parser.parse_args()

    evaluate_model(args)
