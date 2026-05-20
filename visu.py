import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import os

# Custom loss functions (MUST match training script to load model correctly)
def dice_loss(y_true, y_pred):
    """
    Dice loss function for image segmentation.
    """
    smooth = 1e-5
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-5):
    """
    Tversky loss function for balancing precision and recall.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    tp = K.sum(y_true_f * y_pred_f)
    fp = K.sum((1 - y_true_f) * y_pred_f)
    fn = K.sum(y_true_f * (1 - y_pred_f))
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1 - tversky

# Visualization function for segmentation results
def visualize_results():
    # Load preprocessed dataset
    images = np.load("processed_data/images.npy")
    labels = np.load("processed_data/labels.npy")
    labels = np.expand_dims(labels, axis=-1)

    # Load trained model with custom loss functions
    model = load_model(
        "best_model.h5",
        custom_objects={
            "dice_loss": dice_loss,
            "tversky_loss": tversky_loss
        }
    )

    # Split data to get test set (consistent with training)
    split_idx = int(0.8 * len(images))
    x_test = images[split_idx:]
    y_test = labels[split_idx:]

    # Generate model predictions
    y_pred = model.predict(x_test, verbose=0)
    y_pred_bin = (y_pred > 0.5).astype(np.float32)  # Binarize predictions

    # Set high-resolution figure style for paper
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.family'] = 'Arial'

    # Create 3x3 subplot: Original | Ground Truth | Prediction
    num_samples = 3
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    for i in range(num_samples):
        idx = i * 5  # Select sample index
        
        # Show original brain MRI image
        axes[i, 0].imshow(x_test[idx, :, :, 0], cmap="gray")
        axes[i, 0].set_title("Original Image", fontsize=12)
        axes[i, 0].axis("off")

        # Show ground truth tumor label
        axes[i, 1].imshow(y_test[idx, :, :, 0], cmap="gray")
        axes[i, 1].set_title("Ground Truth", fontsize=12)
        axes[i, 1].axis("off")

        # Show model segmentation prediction
        axes[i, 2].imshow(y_pred_bin[idx, :, :, 0], cmap="gray")
        axes[i, 2].set_title("Model Prediction", fontsize=12)
        axes[i, 2].axis("off")

    plt.tight_layout()

    # Save the figure
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/segmentation_comparison.png", bbox_inches="tight")
    plt.close()

    print(" Visualization saved to results/segmentation_comparison.png")

if __name__ == "__main__":
    visualize_results()
