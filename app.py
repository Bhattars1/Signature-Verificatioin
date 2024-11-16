import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import glob
import os
import cv2

from preprocessing.preprocessing import load, extract_blocks, rotate_blocks, classify_blocks
from utils.utils import consistency, split_data, create_dataloader
from models.model import vgg19
from utils.utils import utils, accuracy
from models.training import train
from models.evaluation import evaluate

device = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.ToTensor()
    
def LoadAndPreprocess():
    # Importing the data
    path_1 = "data/Dataset/Sample_12"
    path_2 = "data/Dataset/Sample_15"
    forgeries, original = load(path_1)

    # Preprocessing
    print("Starting preprocessing...")

    # Extracting blocks
    original_block = extract_blocks(original)
    forgeries_block = extract_blocks(forgeries)
        
    # Rotating the images
    rotated_original = rotate_blocks(original_block, 20)
    rotated_forgeries = rotate_blocks(forgeries_block, 20)

    # Filter out invalid images based on pixel intensity
    valid_original = classify_blocks(rotated_original)
    valid_forgeries = classify_blocks(rotated_forgeries)

    # Return preprocessed images
    return valid_original, valid_forgeries

# The model
model = vgg19()  # Instantiate the model

# Loss function, optimizer and accuracy function
loss_fn, optimizer = utils(model=model)

if __name__ == "__main__":
    valid_original, valid_forgeries = LoadAndPreprocess()
    images, labels = consistency(valid_original, valid_forgeries)
    images = [transform(image) for image in images]
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = split_data(images, labels)
    train_dataloader, test_dataloader = create_dataloader(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)    
    train(epochs=1, train_dataloader=train_dataloader, optimizer=optimizer, loss_fn=loss_fn)
    model_save_path = 'models/vgg19_trained_model.pth'
    torch.save(model.state_dict(), model_save_path)
    evaluate(dataloader=test_dataloader, loss_fn=loss_fn)
