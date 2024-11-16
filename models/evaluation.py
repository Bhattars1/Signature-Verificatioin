import torch
from models.model import vgg19
from utils.utils import accuracy, create_dataloader
from tqdm import tqdm

# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the trained model
vgg19 = vgg19()
model_path = 'models/vgg19_trained_model.pth'
vgg19.load_state_dict(torch.load(model_path, map_location=device))
vgg19.to(device)
vgg19.eval()  # Set model to evaluation mode

def evaluate(dataloader, loss_fn):
    """
    Evaluate the model on a given dataloader.

    Args:
        dataloader: DataLoader object for validation or test data.
        loss_fn: Loss function used for evaluation.

    Returns:
        avg_loss: Average loss across all batches.
        avg_accuracy: Average accuracy across all batches.
    """
    total_loss = 0.0
    total_accuracy = 0.0
    total_batches = 0
    print("Evaluating...")
    with torch.no_grad():  # Disable gradient computation
        for X_batch, y_batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs = vgg19(X_batch)

            # Calculate loss
            loss = loss_fn(outputs, y_batch)

            # Calculate accuracy
            acc = accuracy(outputs, y_batch)

            # Accumulate loss and accuracy
            total_loss += loss.item()
            total_accuracy += acc
            total_batches += 1

    # Calculate average loss and accuracy
    avg_loss = total_loss / total_batches
    avg_accuracy = total_accuracy / total_batches

    print(f"Evaluation Results - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2f}%")
    return avg_loss, avg_accuracy