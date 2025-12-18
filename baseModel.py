import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt 
import os
import numpy as np
from sklearn.utils import class_weight 

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the Dataset Root
DATASET_ROOT = os.path.join(SCRIPT_DIR, "prcessed_dataset_05_12_2025_pt2")

TRAIN_DIR = os.path.join(DATASET_ROOT, "train")

# CHANGED: Pointing to the 'test' folder
TEST_DIR = os.path.join(DATASET_ROOT, "test") 

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

# --- FOLDER CHECK ---
if not os.path.exists(TRAIN_DIR):
    print(f"ERROR: Training folder not found at: {TRAIN_DIR}")
    exit()
if not os.path.exists(TEST_DIR):
    print(f"ERROR: Testing folder not found at: {TEST_DIR}")
    exit()

print(f"Training on data from: {TRAIN_DIR}")
print(f"Testing on data from: {TEST_DIR}")

# --- LOAD DATA ---
print("Loading Training Data...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

print("Loading Testing Data...")
# CHANGED: Variable name is now 'test_ds'
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical' 
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print(f"--- CLASS ORDER: {class_names} ---")

# --- CALCULATE CLASS WEIGHTS ---
print("Computing class weights...")
train_labels = []
for images, labels in train_ds:
    train_labels.extend(np.argmax(labels.numpy(), axis=1))

class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = dict(enumerate(class_weights_array))
print(f"Weights Applied: {class_weights}")

# --- OPTIMIZATION ---
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- AUGMENTATION ---
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.1), 
])

# --- BUILD MODEL ---
def build_model():
    base_model = EfficientNetB0(
        include_top=False, 
        weights="imagenet", 
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)    
    x = base_model(x, training=False) 
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model

print("Building Model...")
model = build_model()

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# --- TRAINING ---
print("Starting Training...")
# NOTE: 'val_loss' is an internal Keras keyword. It MUST stay 'val_loss' 
# even though we are feeding it testing data.
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=test_ds, # CHANGED: Passing test_ds here
    epochs=EPOCHS,
    callbacks=[early_stop],
    class_weight=class_weights
)

# --- SAVE MODEL ---
save_path = os.path.join(SCRIPT_DIR, 'soil_classifier_18_12_2025.keras')
print(f"Saving Model to {save_path}...")
model.save(save_path)

# --- PLOTTING ---
# NOTE: Keras saves metrics as 'val_accuracy' and 'val_loss' regardless of variable names
acc = history.history['accuracy']
test_acc = history.history['val_accuracy'] # We assign it to 'test_acc' variable
loss = history.history['loss']
test_loss = history.history['val_loss']    # We assign it to 'test_loss' variable
epochs_range = range(len(acc))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Accuracy')
plt.plot(epochs_range, test_acc, label='Test Accuracy') # Label on graph
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, test_loss, label='Test Loss') # Label on graph
plt.legend()
plt.title('Loss')

plot_path = os.path.join(SCRIPT_DIR, 'training_history.png')
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")
print("Done!")
