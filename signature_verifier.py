import torch
from torchvision import transforms
from PIL import Image
from models.model import vgg19
from preprocessing.preprocessing import extract_blocks, rotate_blocks, classify_blocks
import tkinter as tk
from tkinter import filedialog
import cv2
from tqdm import tqdm

# Define device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Recreate the model architecture
vgg19_model = vgg19()

# Load the trained weights
trained_model_path = 'models/vgg19_trained_model.pth'
vgg19_model.load_state_dict(torch.load(trained_model_path, map_location=device, weights_only=True))
vgg19_model.to(device)
vgg19_model.eval()

# Define preprocessing function
def PreProcess(image_path):
    image = cv2.imread(image_path)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_list = [image]
    image_list = extract_blocks(image_list)
    image_list = rotate_blocks(image_list)
    image_list = classify_blocks(image_list)
    new_image_list = []
    transform = transforms.ToTensor()
    for image in image_list:
        image = transform(image)
        new_image_list.append(image)
    return new_image_list


# Function for inference and classification
def infer_and_classify(image):
    """
    This function preprocesses the image, passes it through the model, 
    and summarizes the binary classification results.
    """
    # Preprocess the image
    preprocessed_images = PreProcess(image)
    # Create a dictionary to hold the counts of binary classifications
    output_counts = {}

    # Loop through each preprocessed image and make predictions
    for img in tqdm(preprocessed_images[:1]):
        # Assuming `img` is a tensor, move it to the appropriate device
        img = img.to(device)

        # Add a batch dimension if the image is single
        if len(img.shape) == 3:  
            img = img.unsqueeze(0)

        # Pass the image through the model
        with torch.no_grad():
            output = vgg19_model(img)
            print(f"The output: {output}")
            prediction = torch.argmax(output, dim=1).item()
            print(f"The prdiction{prediction}")
        # Update the count in the dictionary
        if prediction not in output_counts:
            output_counts[prediction] = 0
        output_counts[prediction] += 1
    key_replacements = {1: "Matched", 0: 'UnMatched'}
    # Replace keys
    for old_key, new_key in key_replacements.items():
        if old_key in output_counts:
            output_counts[new_key] = output_counts.pop(old_key)

    return output_counts


if __name__ == "__main__":
    # Image path to check
    image_path = "data\Dataset\Sample_12\original_12_2.png"
    if image_path:  
        # Infer and classify using the model
        result = infer_and_classify(image_path)

        # Display the results
        print("Classification Results:", result)
    else:
        print("No image selected.")