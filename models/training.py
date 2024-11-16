import torch
from models.model import vgg19
from tqdm import tqdm
from utils.utils import utils, accuracy
from utils.utils import create_dataloader

device = "cuda" if torch.cuda.is_available() else "cpu"

vgg19 = vgg19()
def train(epochs, train_dataloader, optimizer, loss_fn):
    num_epochs = epochs
    train_losses = []  # Stores average loss per epoch
    train_accuracies = []  # Stores average accuracy per epoch

    # Training loop
    vgg19.train()
    print("Strating training")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        total_batches = 0

        # Wrap the data loader with tqdm for batch-level tracking
        for batch_idx, (X_batch, y_batch) in enumerate(tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", unit="batch")):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = vgg19(X_batch)

            # Calculate loss
            loss = loss_fn(outputs, y_batch)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            acc = accuracy(outputs, y_batch)

            # Accumulate loss and accuracy
            epoch_loss += loss.item()
            epoch_accuracy += acc
            total_batches += 1

            # Display results every 500 batches
            if (batch_idx + 1) % 500 == 0:
                avg_loss = epoch_loss / total_batches
                avg_accuracy = epoch_accuracy / total_batches
                print(f'Batch [{batch_idx + 1}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2f}%')

        # Calculate and print average loss and accuracy for the epoch
        avg_loss = epoch_loss / total_batches
        avg_accuracy = epoch_accuracy / total_batches
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2f}%')

        # Store metrics for plotting
        train_losses.append(avg_loss)
        train_accuracies.append(avg_accuracy)