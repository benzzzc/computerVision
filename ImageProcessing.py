import cv2
import os
import numpy as np

INPUT_FOLDER = "raw_photos"
OUTPUT_FOLDER = "processed_dataset"
TARGET_SIZE = 224

def process_image_zero_waste(image_path, filename):
    img = cv2.imread(image_path)
    if img is None: return

    h, w, _ = img.shape

    # --- THE UPGRADE: "Smart Padding" ---
    # Instead of just checking if it's too small, we check the REMAINDER.
    # We want the image size to be a perfect multiple of 224 (e.g., 224, 448, 672...)
    
    # 1. Calculate how many pixels are missing to reach the next multiple of 224
    # The % symbol is "Modulo" (Remainder)
    remainder_h = h % TARGET_SIZE
    remainder_w = w % TARGET_SIZE
    
    pad_h = 0
    pad_w = 0
    
    # If there is a remainder, calculate how much to add to reach 224
    if remainder_h > 0:
        pad_h = TARGET_SIZE - remainder_h
        
    if remainder_w > 0:
        pad_w = TARGET_SIZE - remainder_w

    # 2. Apply the padding to the Bottom and Right edges only
    # We use Mirror Reflection so the AI just sees "more texture"
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(
            img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
        )

    # Update dimensions (Now the image is a perfect multiple of 224)
    h, w, _ = img.shape

    # --- STEP 2: SLIDING WINDOW (Standard) ---
    STRIDE = 224
    count = 0
    
    for y in range(0, h, STRIDE):        # Note: Simplified loop range
        for x in range(0, w, STRIDE):    # Note: Simplified loop range
            
            crop = img[y:y+TARGET_SIZE, x:x+TARGET_SIZE]
            
            # Safety check: Ensure crop is exactly 224x224
            if crop.shape[0] == TARGET_SIZE and crop.shape[1] == TARGET_SIZE:
                augment_and_save(crop, filename, f"patch{count}")
                count += 1

def augment_and_save(image, base_name, suffix):
    def save(name, img):
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, name), img)
    
    save(f"{base_name}_{suffix}_orig.jpg", image)
    save(f"{base_name}_{suffix}_flipH.jpg", cv2.flip(image, 1))
    save(f"{base_name}_{suffix}_flipV.jpg", cv2.flip(image, 0))
    save(f"{base_name}_{suffix}_flipHV.jpg", cv2.flip(image, -1))

# --- MAIN ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.jpg', '.png'))]
    print(f"Processing {len(files)} images with Zero Waste mode...")

    for file_name in files:
        process_image_zero_waste(
            os.path.join(INPUT_FOLDER, file_name), 
            os.path.splitext(file_name)[0]
        )
    print("Done!")
