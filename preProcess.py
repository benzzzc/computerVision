import cv2
import os
import numpy as np

# CONSTANTS
INPUT_FOLDER = "raw_photos"         
OUTPUT_FOLDER = "processed_dataset" 
TARGET_SIZE = 224                   

def process_image_universal(image_path, filename, destination_folder):
    img = cv2.imread(image_path)
    if img is None: return

    h, w, _ = img.shape
    
    # --- STEP 1: PADDING ---
    pad_h = max(0, TARGET_SIZE - h)
    pad_w = max(0, TARGET_SIZE - w)
    
    if pad_h > 0 or pad_w > 0:
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_REFLECT_101
        )
    
    h, w, _ = img.shape

    # --- STEP 2: SLIDING WINDOW ---
    STRIDE = 224
    count = 0
    
    for y in range(0, h - TARGET_SIZE + 1, STRIDE):
        for x in range(0, w - TARGET_SIZE + 1, STRIDE):
            crop = img[y:y+TARGET_SIZE, x:x+TARGET_SIZE]
            
            # Send to augment function with the correct sub-folder
            augment_and_save(crop, filename, f"patch{count}", destination_folder)
            count += 1

def augment_and_save(image, base_name, suffix, destination_folder):
    def save(name, img):
        # Save into the specific class folder (e.g. processed_dataset/Clay)
        cv2.imwrite(os.path.join(destination_folder, name), img)

    save(f"{base_name}_{suffix}_orig.jpg", image)
    save(f"{base_name}_{suffix}_flipH.jpg", cv2.flip(image, 1))
    save(f"{base_name}_{suffix}_flipV.jpg", cv2.flip(image, 0))
    save(f"{base_name}_{suffix}_flipHV.jpg", cv2.flip(image, -1))

def main():
    print("Starting pre-processing...")

    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Folder '{INPUT_FOLDER}' not found.")
        return

    # Loop through Clay, Sand, Loam folders
    for class_name in os.listdir(INPUT_FOLDER):
        
        class_input_path = os.path.join(INPUT_FOLDER, class_name)
        class_output_path = os.path.join(OUTPUT_FOLDER, class_name)

        # Skip random files, only look for folders
        if not os.path.isdir(class_input_path): continue

        # Make output folder (e.g. processed_dataset/Clay)
        if not os.path.exists(class_output_path):
            os.makedirs(class_output_path)
            
        print(f"Processing Class: {class_name}...")

        files = [f for f in os.listdir(class_input_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        for file_name in files:
            process_image_universal(
                os.path.join(class_input_path, file_name), 
                os.path.splitext(file_name)[0],
                class_output_path
            )

    print("All Done! Check your 'processed_dataset' folder.")

if __name__ == "__main__":
    main()
