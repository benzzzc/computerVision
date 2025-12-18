import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt 
import os
from sklearn.utils import class_weight
import numpy as np

# --- PATH CONFIGURATION ---
# 1. Get the folder where this script lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Define the Dataset Root
# Note: Keeping your spelling 'prcessed' to match previous steps
DATASET_ROOT = os.path.join(SCRIPT_DIR, "prcessed_dataset_05_12_2025_pt2")

# 3. Define Train and Validation/Test folders specifically
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")

# IMPORTANT: Ensure you have a 'test' or 'val' folder in your processed dataset!
# If your folder is named 'val', change "test" to "val" below.
TEST_DIR = os.path.join(DATASET_ROOT, "test") 

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 3 
EPOCHS = 20

# --- VALIDATION CHECK ---
if not os.path.exists(TRAIN_DIR):
    print(f"ERROR: Training folder not found at: {TRAIN_DIR}")
    exit()
if not os.path.exists(TEST_DIR):
    print(f"WARNING: Validation folder not found at: {TEST_DIR}")
    print("Please make sure you have processed your 'test' or 'val' split using the ImageProcessing script.")
    # We exit here because the model needs validation data to run the code below
    exit()

print(f"Training on data from: {TRAIN_DIR}")
print(f"Testing on data from: {TEST_DIR}")

# --- LOAD DATA (No splitting parameters) ---
print("Loading Training Data...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
    # Removed: validation_split, subset
)

print("Load testing data...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
    
)

class_names = train_ds.class_names
print(f"--- CLASS ORDER FOUND: {class_names} ---")
# Only print if we actually have enough classes to avoid index errors
for i, name in enumerate(class_names):
    print(f"{i} = {name}")

#   optised caching
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#   alerady augmented so minimal prcesses
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.1), 
])

#   build model
def build_model():
    base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)    
    x = base_model(x, training=False) 
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model

#   train the model
print("Building Model...")
model = build_model()

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("Starting Training...")
early_stop = EarlyStopping(monitor='training_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

#   save the model
print("Saving Model...")
model.save(os.path.join(SCRIPT_DIR, 'soil_classifier_18_12_2025.keras'))

#   plot the train, validation data
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['testing_accuracy'], label='Testing Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['testing_loss'], label='Testing Loss')
plt.legend()
plt.title('Loss')
plt.savefig(os.path.join(SCRIPT_DIR, 'traningHistory.png')) # Save instead of show for stability
print("Plot saved ")
# plt.show() # Uncomment if running locally with a screen
print("Done!")


#       CLEAN THE CODE, ESPECIALLY FOR CHANGE IN VAL --> TEST 
