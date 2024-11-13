# Image/label consistency, train/test split and prepraring the dataloader

import random
import sklearn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.optim as optim
from models.model import vgg19

device = "cuda" if torch.cuda.is_available() else "cpu"

def consistency(valid_original1, valid_forgeries1):
    images = valid_original1 + valid_forgeries1
    labels = [1] * len(valid_original1)+ [0]*len(valid_forgeries1)

    combined = list(zip(images, labels))
    random.shuffle(combined)
    images, labels = zip(*combined)
    images = list(images)
    labels = list(labels)
    return images, labels


def split_data(images, labels):
    # Splittin 80% of the data into training set and remaining 20% to testing set
    X_train, X_test, y_train, y_test = train_test_split(images,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=42)
    # Sending the data  and labels to the current device to avoid device mismatch error
    X_train = [tensor.to(device) for tensor in X_train]
    y_train = [torch.tensor(tensor).to(device) for tensor in y_train]
    X_test = [tensor.to(device) for tensor in X_test]
    y_test = [torch.tensor(tensor).to(device) for tensor in y_test]

    X_train_tensor = torch.stack(X_train).to(device)
    y_train_tensor = torch.stack(y_train).to(device)
    X_test_tensor = torch.stack(X_test).to(device)
    y_test_tensor = torch.stack(y_test).to(device)
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

def create_dataloader(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor):
    # Create TensorDatasets
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    test_data = TensorDataset(X_test_tensor, y_test_tensor)


    # Setup the batch size 
    BATCH_SIZE = 4

    # Turn datasets into iterables (batches)
    train_dataloader = DataLoader(train_data,  # dataset to turn into iterable
        batch_size=BATCH_SIZE,  # how many samples per batch?
        shuffle=True 
    )

    test_dataloader = DataLoader(test_data,
        batch_size=BATCH_SIZE,
        shuffle=False 
    )
    return train_dataloader, test_dataloader

def utils(model):

    loss_fn = nn.CrossEntropyLoss()
    # Define learning rate and momentum
    learning_rate = 1e-4
    momentum = 0.9

    # Initialize SGD optimizer, only including parameters that require gradients
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum=momentum)
    return loss_fn, optimizer

# Define a accuracy function
def accuracy(outputs, labels):
    # Get predicted class (no need for sigmoid here)
    predictions = outputs.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    return correct / labels.size(0) * 100  # Convert to percentage



