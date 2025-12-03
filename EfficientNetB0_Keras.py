import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping

# --- CONFIGURATION ---
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 3  # Sand, Silt, Clay
EPOCHS = 20

# --- 1. LOAD IMAGES ---
# This automatically reads your folders and turns them into data
print("Loading Training Data...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",              # Your main folder
    validation_split=0.2,   # Use 20% for testing
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical' # 'categorical' for 3 classes
)

print("Loading Validation Data...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# --- 2. SETUP AUGMENTATION ---
# This makes fake versions of your images (flipped, rotated) to help training
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

# --- 3. BUILD THE MODEL ---
def build_model():
    # THIS is where we use the "Engine" without building it from scratch
    # weights="imagenet" downloads the pre-trained brain
    base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Freeze the engine so we don't break it
    base_model.trainable = False

    # Connect our own inputs and outputs
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)  # augment first
    x = base_model(x, training=False) 
    x = layers.GlobalAveragePooling2D()(x) # Flatten the data
    x = layers.Dropout(0.2)(x)             # Prevent overfitting
    
    # The final answer layer (3 classes)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model

# --- 4. TRAIN ---
print("Building Model...")
model = build_model()

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("Starting Training...")
# This stops training if the model stops improving (saves time)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# --- 5. SAVE ---
print("Saving Model...")
model.save('soil_classifier.keras')
print("Done! You can now use the model.")
