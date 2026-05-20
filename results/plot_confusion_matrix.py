import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import os

# Custom loss functions
def dice_loss(y_true, y_pred):
    smooth = 1e-5
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-5):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    tp = K.sum(y_true_f * y_pred_f)
    fp = K.sum((1 - y_true_f) * y_pred_f)
    fn = K.sum(y_true_f * (1 - y_pred_f))
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1 - tversky

def plot_confusion_matrix():
    # Load dataset
    images = np.load("processed_data/images.npy")
    labels = np.load("processed_data/labels.npy")
    labels = np.expand_dims(labels, axis=-1)

    # Split test set
    split_idx = int(0.8 * len(images))
    x_test = images[split_idx:]
    y_test = labels[split_idx:]

    # Load model
    model = load_model("best_model.h5",
                       custom_objects={"dice_loss":dice_loss,"tversky_loss":tversky_loss})
    y_pred = model.predict(x_test, verbose=0)
    y_pred_bin = (y_pred > 0.5).astype(np.float32)

    # Flatten all pixels
    y_true_flat = y_test.flatten()
    y_pred_flat = y_pred_bin.flatten()

    # Calculate confusion matrix
    tp = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
    tn = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
    fp = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
    fn = np.sum((y_true_flat == 1) & (y_pred_flat == 0))

    cm = np.array([[tn, fp], [fn, tp]])

    # Plot setting
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.family'] = 'Arial'
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Background","Tumor"],
                yticklabels=["Background","Tumor"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Pixel-level Confusion Matrix")
    os.makedirs("results", exist_ok=True)
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", bbox_inches="tight")
    plt.close()
    print(" Confusion matrix saved")

if __name__=="__main__":
    plot_confusion_matrix()
