import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import glob
import os
import cv2

from preprocessing.preprocessing import load, extract_blocks, rotate_blocks, classify_blocks, save_images
from utils.utils import consistency, split_data, create_dataloader
from models.model import vgg19
from utils.utils import utils, accuracy
from models.training import train

device = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.ToTensor()

def LoadAndPreprocess(original_path = 'data\valid_images', forgeries_path = 'data\forged_images'):
    ## Importing the data
    path_1 = "data\Dataset\Sample_12"
    path_2 = "data\Dataset\Sample_15"
    forgeries, original = load(path_1)

    ## Preporcessing

    # Extracting blocks
    original_block = extract_blocks(original)
    forgeries_block = extract_blocks(forgeries)
        
    # Rotating the images
    rotated_original = rotate_blocks(original_block, 10)
    rotated_forgeries = rotate_blocks(forgeries_block, 30)

    # Filter out invalid images based on pixel intensity
    valid_original = classify_blocks(rotated_original)
    valid_forgeries = classify_blocks(rotated_forgeries)

    # Save the preprocessed images: "CHECKPOINT"
    save_images(valid_original, original_path, 'valid_original')
    save_images(valid_forgeries, forgeries_path, 'valid_forgery')
    
def LoadAndPreprocess(original_path='data/valid_images', forgeries_path='data/forged_images'):
    def load_images_from_directory(directory):
        image_files = glob.glob(os.path.join(directory, '*.png')) + glob.glob(os.path.join(directory, '*.jpg'))
        images = [cv2.imread(image_path) for image_path in image_files]
        return images if images else None

    # Check if processed images already exist in the specified paths
    original_images = load_images_from_directory(original_path)
    forgery_images = load_images_from_directory(forgeries_path)

    # If both original and forgery images exist, skip preprocessing
    if original_images and forgery_images:
        print("Loaded existing preprocessed images from disk.")
        return original_images, forgery_images

    # Proceed with preprocessing if images are not found
    print("No existing images found, starting preprocessing...")

    # Importing the data
    path_1 = "data/Dataset/Sample_12"
    path_2 = "data/Dataset/Sample_15"
    forgeries, original = load(path_1)

    # Preprocessing
    # Extracting blocks
    original_block = extract_blocks(original)
    forgeries_block = extract_blocks(forgeries)
        
    # Rotating the images
    rotated_original = rotate_blocks(original_block, 10)
    rotated_forgeries = rotate_blocks(forgeries_block, 30)

    # Filter out invalid images based on pixel intensity
    valid_original = classify_blocks(rotated_original)
    valid_forgeries = classify_blocks(rotated_forgeries)

    # # Save the preprocessed images
    # save_images(valid_original, original_path, 'valid_original')
    # save_images(valid_forgeries, forgeries_path, 'valid_forgery')

    # Return preprocessed images
    return valid_original, valid_forgeries

valid_original, valid_forgeries = LoadAndPreprocess()

## Maintain data and labels consistency
images, labels = consistency(valid_original, valid_forgeries)

## Convert all images to tensors
images = [transform(image) for image in images]

## Train Test Split and dataloader
X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = split_data(images, labels)
train_dataloader, test_dataloader = create_dataloader(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)

# The model
vgg19 = vgg19()

# Loss function, optimizer and accuracy function
loss_fn, optimizer = utils(model=vgg19)

train(epochs=1, train_dataloader=train_dataloader, optimizer=optimizer, loss_fn=loss_fn)

if __name__ == "__main__":
    valid_original, valid_forgeries = LoadAndPreprocess()
    images, labels = consistency(valid_original, valid_forgeries)
    images = [transform(image) for image in images]
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = split_data(images, labels)
    train_dataloader, test_dataloader = create_dataloader(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
    vgg19 = vgg19()
    loss_fn, optimizer = utils(model=vgg19)
    train(epochs=1, train_dataloader=train_dataloader, optimizer=optimizer, loss_fn=loss_fn)