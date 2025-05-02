import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_model(factory, model_name, model_file, random_model=False, device="cpu"):
    """Load model from a file

    Parameters
    ----------
    factory : 
        Name of the model factory
    model_name : str
        Name of the model
    model_file : str
        file to load
    random_model : bool, optional
        Use an initialized model, by default False
    device : str, optional
        device to run the model on, by default "cpu"

    Returns
    -------
    BaseModel
        Model to use
    """
    model_to_evaluate = factory.get_model(model_name).to(device)

    # Load the model
    if random_model == False:
        model_file = f'./models/files/{model_file}.pth'
        print(f'Loading model {model_file}')

        model_to_evaluate.load_state_dict(torch.load(model_file))

    model_to_evaluate.eval()  # Set to evaluation mode

    return model_to_evaluate

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