import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

def create_model():
    # Backbone
    base = MobileNetV2(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False # Initial training with frozen backbone

    # Custom Head
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)

    # Classification Branch
    class_output = Dense(128, activation="relu")(x)
    class_output = Dropout(0.5)(class_output)
    class_output = Dense(3, activation="softmax", name="class_output")(class_output)

    # Bounding Box Regression Branch
    bbox_output = Dense(128, activation="relu")(x)
    bbox_output = Dropout(0.5)(bbox_output)
    bbox_output = Dense(4, activation="sigmoid", name="bbox_output")(bbox_output)

    model = Model(inputs=base.input, outputs=[class_output, bbox_output])
    return model

def load_data():
    data = np.load("data/dataset.npz")
    return (data["X_train"], data["yb_train"], data["yl_train"]), \
           (data["X_val"], data["yb_val"], data["yl_val"])

def augment(image, bbox, label):
    # Random horizontal flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        # Flip bbox: [ymin, xmin, ymax, xmax] in TF? 
        # Dataset uses [xmin, ymin, xmax, ymax].
        # xmin -> 1 - xmax
        # xmax -> 1 - xmin
        ymin = bbox[1]
        ymax = bbox[3]
        xmin = 1.0 - bbox[2]
        xmax = 1.0 - bbox[0]
        bbox = tf.stack([xmin, ymin, xmax, ymax])
    
    # Color Jitter
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, (label, bbox) # Tuple matching model outputs

def preprocess(image, bbox, label):
    return image, (label, bbox)

if __name__ == "__main__":
    if not os.path.exists("model"):
        os.makedirs("model")

    (X_train, yb_train, yl_train), (X_val, yb_val, yl_val) = load_data()
    print(f"Training on {len(X_train)} samples, Validating on {len(X_val)} samples.")

    # Create Datasets
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, yb_train, yl_train))
    train_ds = train_ds.shuffle(len(X_train)).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, yb_val, yl_val))
    val_ds = val_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Build Model
    model = create_model()
    
    losses = {
        "class_output": "categorical_crossentropy",
        "bbox_output": "mse"
    }
    
    # Weight the losses: Bbox loss is often small (MSE of 0-1 values), so scale it up
    loss_weights = {
        "class_output": 1.0,
        "bbox_output": 10.0 # Heuristic to balance classification and regression
    }

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=losses,
        loss_weights=loss_weights,
        metrics={
            "class_output": "accuracy",
            "bbox_output": "mse"
        }
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint("model/face_mask_detector.keras", monitor="val_loss", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, verbose=1, restore_best_weights=True)
    ]

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print("Training complete. Model saved.")
