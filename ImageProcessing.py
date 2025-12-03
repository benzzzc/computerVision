# CODE TO PROCESS IMAGES FOR EFFICIENTNET B0
# This script handles three scenarios:
# 1. Too Small: Uses mirror padding to reach 224x224 without stretching/blurring.
# 2. Too Large: Uses a sliding window to slice into multiple 224x224 patches.
# 3. Weird Shapes (e.g., 166x800): Pads the short side, then slices the long side.
# Result: High-quality texture data with 4x Augmentation (flips).

# may need to CLAHE

# restrcuture the code so it is cleaner and easy to understand, the comments are too much

import cv2
import os
import numpy as np

# Configure the folder destination
INPUT_FOLDER = "raw_photos"         
OUTPUT_FOLDER = "processed_dataset" 
TARGET_SIZE = 224                   

# This acts as a pipeline: Input Image -> Pad if needed -> Crop whatever is left
def process_image_universal(image_path, filename):
    img = cv2.imread(image_path)
    if img is None: return
    h, w, _ = img.shape
    
    # Calculate how many pixels are missing (max(0, ...) ensures no negative numbers if dimension is large)
    pad_h = max(0, TARGET_SIZE - h)
    pad_w = max(0, TARGET_SIZE - w)
    
    # If padding is required (pixels missing > 0)
    if pad_h > 0 or pad_w > 0:
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)
        
        # cv2.BORDER_REFLECT_101 mirrors the edge pixels to fill the gap.
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_REFLECT_101
        )
    
    # Update the dimensions variables (h, w) because the image might be bigger now after padding
    h, w, _ = img.shape

    # Now that we guarantee the image is at least 224x224, we slice it up.
    # This turns one panoramic image (e.g., 224x800) into multiple squares (224x224).
    
    STRIDE = 224 # Move exactly one block width at a time (no overlap)
    count = 0   
    
    # Like the code for image segementation
    # https://pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
    for y in range(0, h - TARGET_SIZE + 1, STRIDE):
        for x in range(0, w - TARGET_SIZE + 1, STRIDE):
            crop = img[y:y+TARGET_SIZE, x:x+TARGET_SIZE]
            augment_and_save(crop, filename, f"patch{count}")
            count += 1
s
# Takes one finished 224x224 patch and creates 3 extra copies by flipping.
def augment_and_save(image, base_name, suffix):
    # Save the resived image as OG
    cv2.imwrite(f"{OUTPUT_FOLDER}/{base_name}_{suffix}_orig.jpg", image)
    
    # Flip OG horizontally and save
    cv2.imwrite(f"{OUTPUT_FOLDER}/{base_name}_{suffix}_flipH.jpg", cv2.flip(image, 1))
    
    # Flip OG vertically and save
    cv2.imwrite(f"{OUTPUT_FOLDER}/{base_name}_{suffix}_flipV.jpg", cv2.flip(image, 0))
    
    # # Flip OG around and save
    cv2.imwrite(f"{OUTPUT_FOLDER}/{base_name}_{suffix}_flipHV.jpg", cv2.flip(image, -1))

# Like the main loop in c ^^ is the function definitions

# Creates output folder if it does not exist so the script doesn't crash on save
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Iterates through every file in the raw_photos folder
for file_name in os.listdir(INPUT_FOLDER):
    # Checks file extension to ensure we only process images (ignores .DS_Store or text files)
    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        
        # Joins path (e.g., "raw_photos/sand.jpg") and strips extension for naming
        process_image_universal(
            os.path.join(INPUT_FOLDER, file_name), 
            os.path.splitext(file_name)[0]
        )

# Final confirmation message
print("Processing complete! Images padded (if small), cropped (if large), and flipped.")
