import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from tqdm import tqdm
import numpy as np
from .utils import reshape_xXy

def plot_data_imabalance(data):
    """Plot the data imbalance based on the Label column

    """
    grouped_sum = data.groupby('Label').size()

    plt.figure(figsize=(3,3))
    
    ax = sns.barplot(data=grouped_sum)
    # Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.show()

def graph_attack_packets_headers(data):
    """Plotting the attack packets as an image

    Parameters
    ----------
    data : Series
        data packet
    """
    categories = data.Attack.unique()

    columns = [f'Header_bit_{i}' for i in range(1,145)]  + ['Attack']
    grouped_mean = data.loc[:,columns].groupby('Attack').mean()

    col_num = 4
    row_num = int(len(categories) / col_num) + 1
    fig, axes = plt.subplots(row_num,col_num, figsize=(10,6))

    # Flatten axes array for easy iteration
    axes_flat = axes.flatten()

    for i, category in enumerate(categories):
        ax = axes_flat[i]
        sns.heatmap(reshape_xXy(grouped_mean.loc[category,:]),
                ax=ax,
                cmap='YlOrBr',
                square=True,
            cbar=False)
        ax.set_title(category)

    plt.show()

def graph_attack_packets_entropy(data):
    """Plot the average of the data for the entropy information
    """
    categories = data.Attack.unique()

    columns = [f'Entropy_{i}' for i in range(1,11)] + [ 'Attack']
    grouped_mean = data.loc[:,columns].groupby('Attack').mean()


    col_num = 2
    row_num = int(len(categories) / col_num) + 1
    fig, axes = plt.subplots(row_num,col_num, figsize=(len(categories) * 2,6))

    axes_flat = axes.flatten()

    for i, category in enumerate(categories):
        ax = axes_flat[i]
        sns.heatmap(reshape_xXy(grouped_mean.loc[category,:],10,1).T,
                ax=ax,
                cmap='YlOrBr',
                square=True,
            cbar=False)
        ax.set_title(category)

    plt.show()


def plot_attention_map(benign, malicious):
    """Plot the attention map

    Parameters
    ----------
    benign : 
        Average benign heatmap
    malicious : 
        Average malicious heatmap
    """
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the first array
    im1 = ax1.imshow(benign, cmap='viridis')
    ax1.set_title('Benign')
    fig.colorbar(im1, ax=ax1)

    # Plot the second array
    im2 = ax2.imshow(malicious, cmap='viridis')
    ax2.set_title('Malicious')
    fig.colorbar(im2, ax=ax2)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model_to_evaluate, dataloader, device="cpu"):
    """Plot the confusion matrix
    """
    y_true = []
    y_pred = []

    with torch.no_grad():  # Disable gradient tracking
        for inputs, labels in tqdm(dataloader):  # Assuming you have a DataLoader
            outputs = model_to_evaluate(inputs.to(device).float())
            predicted_probs = torch.sigmoid(outputs)  # Convert logits to probabilities
            predicted_labels = (predicted_probs > 0.5).float()  # Threshold at 0.5
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted_labels.cpu().numpy())

    

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Predicted 0", "Predicted 1"], 
                yticklabels=["Actual 0", "Actual 1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    return
