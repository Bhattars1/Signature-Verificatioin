# Device agnostic code
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

from PIL import Image
import numpy as np
import os
import cv2
     
def load(path):
  forgeries = []
  original = []
  for file_name in os.listdir(path):
    file_path = os.path.join(path, file_name)
    image = cv2.imread(file_path)
    try:
      image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
      print("invalid file format")
    if "forgeries" in file_name.lower():
      forgeries.append(image)
    elif "original" in file_name.lower():
      original.append(image)
    else:
      pass
  return forgeries, original


def extract_blocks(image_list, window_size = (224,224), step_size = 20):
    """
    Extract overlapping blocks from a list of images using a sliding window approach
    and adjust brightness of the blocks.

    Parameters:
    - images: List of images (as numpy arrays).
    - window_size: Tuple (width, height) of the sliding window.
    - step_size: Number of pixels to shift the window each time.

    Returns:
    - List of extracted blocks with brightness adjusted.
    """
    blocks = []
    brightness_factor = 1.075
    # Iterates through each image
    for image in image_list:
      height, width = image.shape[:2]

      # Slide the window across image
      for y in range(0, height-window_size[1]+1, step_size):
        for x in range(0, width-window_size[0]+1, step_size):
          block = image[y:y+window_size[1], x:x + window_size[0]]
          # Adjust brightness: multiply each pixel by the brightness factor
          brightened_block = cv2.multiply(block, np.array([brightness_factor, brightness_factor, brightness_factor]))

          # Clip the values to ensure they are in the valid range [0, 255]
          brightened_block = np.clip(brightened_block, 0, 255).astype(np.uint8)
          blocks.append(brightened_block)
    return blocks

def rotate_blocks(blocks, angle):
    """
    Rotate each sub-image block by a predefined angle until a full 360 degrees rotation is done.

    Parameters:
    - blocks: List of sub-image blocks (as numpy arrays).
    - angle: The angle (in degrees) to rotate each block.

    Returns:
    - List of rotated blocks.
    """
    rotated_blocks = []

    for block in blocks:
        # Get the center of the block for rotation
        center = (block.shape[1] // 2, block.shape[0] // 2)

        # Rotate the block for each angle from 0 to 360 degrees
        for i in range(0, 360, angle):
            # Get the rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, i, 1.0)

            # Perform the rotation
            rotated_block = cv2.warpAffine(block, rotation_matrix, (block.shape[1], block.shape[0]))

            # Append the rotated block to the list
            rotated_blocks.append(rotated_block)

    return rotated_blocks

def classify_blocks(blocks, threshold=250, valid_percentage=0.075):
    """
    Classify each sub-image block as valid or invalid based on pixel intensity.

    Parameters:
    - blocks: List of sub-image blocks (as numpy arrays).
    - threshold: Grayscale value threshold for valid pixels.
    - valid_percentage: Percentage threshold for considering a block valid.

    Returns:
    - List of valid blocks.
    """
    valid_blocks = []

    for block in blocks:

        # Count valid pixels that exceed the threshold
        valid_pixel_count = np.sum(block > threshold)
        total_pixel_count = block.size

        # Calculate the percentage of valid pixels
        valid_pixel_ratio = valid_pixel_count / total_pixel_count

        # Check if the ratio exceeds the specified valid percentage
        if valid_pixel_ratio > valid_percentage:
            valid_blocks.append(block)  # Keep valid blocks

    return valid_blocks

def save_images(images, save_dir, prefix):
    """
    Save images to the specified directory with a given prefix.

    Parameters:
    - images: List of images (as numpy arrays).
    - save_dir: Directory where the images will be saved.
    - prefix: Prefix for the image filenames.
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx, img in enumerate(images):
        # Convert to uint8 if necessary and save the image
        img_to_save = (img * 255).astype(np.uint8)  # Assuming img is normalized between 0 and 1
        filename = os.path.join(save_dir, f"{prefix}_{idx}.png")
        cv2.imwrite(filename, img_to_save)

