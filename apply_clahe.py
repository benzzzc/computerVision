import cv2
import os
import numpy as np

# CONFIGURATION
INPUT_FOLDER = "processed_dataset"   # Your existing 224x224 patches
OUTPUT_FOLDER = "clahe_dataset"      # Where the enhanced versions go

def apply_clahe_to_image(image):
    # 1. Convert to LAB Color Space
    # We want to enhance Lightness (Texture) but keep Color (A/B) accurate.
    # If we did this on RGB, the colors would look "radioactive" and fake.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # 2. Split the channels
    l, a, b = cv2.split(lab)
    
    # 3. Apply CLAHE to the L-channel (Lightness)
    # clipLimit=2.0 -> Threshold for contrast limiting (Higher = more granular texture)
    # tileGridSize=(8,8) -> Size of the local grid for histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    
    # 4. Merge back and convert to BGR (OpenCV standard)
    lab_merged = cv2.merge((l_enhanced, a, b))
    final_img = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)
    
    return final_img

def main():
    print("Starting CLAHE enhancement...")

    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Could not find '{INPUT_FOLDER}'. Run your preprocessing script first.")
        return

    # Loop through Clay, Sand, Loam folders
    for class_name in os.listdir(INPUT_FOLDER):
        
        input_class_path = os.path.join(INPUT_FOLDER, class_name)
        output_class_path = os.path.join(OUTPUT_FOLDER, class_name)

        # Skip files, only look for folders
        if not os.path.isdir(input_class_path): continue

        # Create output folder
        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)
            
        print(f"Enhancing Class: {class_name}...")

        # Get all images
        files = [f for f in os.listdir(input_class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        for file_name in files:
            # Read
            img_path = os.path.join(input_class_path, file_name)
            img = cv2.imread(img_path)
            
            if img is None: continue
            
            # Enhance
            enhanced_img = apply_clahe_to_image(img)
            
            # Save (We use the same filename)
            save_path = os.path.join(output_class_path, file_name)
            cv2.imwrite(save_path, enhanced_img)

    print(f"Done! Use the '{OUTPUT_FOLDER}' folder for training your model.")

if __name__ == "__main__":
    main()
