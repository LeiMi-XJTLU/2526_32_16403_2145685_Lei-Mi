import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K

# Data augmentation to prevent overfitting
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)

# Dice Loss (for class imbalance)
def dice_loss(y_true, y_pred):
    smooth = 1e-5
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    loss = 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return loss

# Simplified UNet with Dropout & Smaller Channels (Reduce Complexity)
def build_transunet(input_size=(224, 224, 3)):
    inputs = Input(input_size)

    # Encoder (smaller channels to reduce overfitting)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Dropout(0.2)(c1)  # Dropout for regularization
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = Dropout(0.2)(c2)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Decoder
    u4 = UpSampling2D((2, 2))(p3)
    u4 = concatenate([u4, c3])
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(u4)

    u5 = UpSampling2D((2, 2))(c4)
    u5 = concatenate([u5, c2])
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(u5)

    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c1])
    c6 = Conv2D(32, (3, 3), activation='relu', padding='same')(u6)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c6)

    model = Model(inputs, outputs)
    # Use Dice Loss to solve class imbalance
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=dice_loss, metrics=['accuracy'])
    return model

def load_data():
    images = np.load("processed_data/images.npy")
    labels = np.load("processed_data/labels.npy")
    labels = np.expand_dims(labels, axis=-1)
    
    split_idx = int(0.8 * len(images))
    x_train, x_test = images[:split_idx], images[split_idx:]
    y_train, y_test = labels[:split_idx], labels[split_idx:]
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_data()
    model = build_transunet()
    model.summary()

    # Save best model
    checkpoint = ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True, verbose=1)

    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)

    print("\n Training with anti-overfitting strategies...")
    model.fit(
        datagen.flow(x_train, y_train, batch_size=4),
        validation_data=(x_test, y_test),
        epochs=30,
        callbacks=[checkpoint, early_stop]
    )

    print("\n Training finished! Best model saved.")

# Save training history to a file
import pickle
with open("training_history.pkl", "wb") as f:
    pickle.dump(model.history.history, f)
print(" Training history saved to training_history.pkl")
