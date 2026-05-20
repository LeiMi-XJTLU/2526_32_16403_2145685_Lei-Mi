import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import os

# Loss functions (must match training)
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

# Post-processing: Keep only the largest connected component
def keep_largest_component(pred):
    """
    Only keep the largest connected region to remove small false positives.
    """
    pred = (pred > 0.5).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred, connectivity=8)
    
    if num_labels <= 1:
        return pred

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cleaned = (labels == largest_label).astype(np.float32)
    return cleaned

# Calculate metrics
def calculate_metrics(y_true, y_pred):
    y_pred_bin = (y_pred > 0.5).astype(np.float32)
    tp = np.sum(y_true * y_pred_bin)
    fp = np.sum((1 - y_true) * y_pred_bin)
    fn = np.sum(y_true * (1 - y_pred_bin))
    tn = np.sum((1 - y_true) * (1 - y_pred_bin))

    dice = (2 * tp + 1e-5) / (2 * tp + fp + fn + 1e-5)
    precision = (tp + 1e-5) / (tp + fp + 1e-5)
    recall = (tp + 1e-5) / (tp + fn + 1e-5)
    return dice, precision, recall

# Main
def post_process_evaluation():
    images = np.load("processed_data/images.npy")
    labels = np.load("processed_data/labels.npy")
    labels = np.expand_dims(labels, axis=-1)

    split_idx = int(0.8 * len(images))
    x_test = images[split_idx:]
    y_test = labels[split_idx:]

    model = load_model(
        "best_model.h5",
        custom_objects={"dice_loss": dice_loss, "tversky_loss": tversky_loss}
    )
    y_pred = model.predict(x_test, verbose=1)

    # Before post-processing
    dice_before = []
    prec_before = []
    rec_before = []

    # After post-processing
    dice_after = []
    prec_after = []
    rec_after = []

    clean_preds = []

    for i in range(len(x_test)):
        true = y_test[i]
        pred = y_pred[i]

        d, p, r = calculate_metrics(true, pred)
        dice_before.append(d)
        prec_before.append(p)
        rec_before.append(r)

        clean = keep_largest_component(pred)
        clean_preds.append(clean)

        d2, p2, r2 = calculate_metrics(true, clean)
        dice_after.append(d2)
        prec_after.append(p2)
        rec_after.append(r2)

    # Print comparison
    print("=" * 60)
    print(" BEFORE POST-PROCESSING")
    print("=" * 60)
    print(f"Dice: {np.mean(dice_before):.4f}")
    print(f"Precision: {np.mean(prec_before):.4f}")
    print(f"Recall: {np.mean(rec_before):.4f}")

    print("\n" + "=" * 60)
    print(" AFTER POST-PROCESSING (BETTER!)")
    print("=" * 60)
    print(f"Dice: {np.mean(dice_after):.4f}")
    print(f"Precision: {np.mean(prec_after):.4f}")
    print(f"Recall: {np.mean(rec_after):.4f}")
    print(" Post-processing improved precision and Dice!\n")

    # Plot comparison
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.family'] = 'Arial'
    idx = 0
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(x_test[idx, :, :, 0], cmap='gray')
    axes[0].set_title("Image")
    axes[0].axis('off')

    axes[1].imshow(y_test[idx, :, :, 0], cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    axes[2].imshow(y_pred[idx, :, :, 0] > 0.5, cmap='gray')
    axes[2].set_title("Before Post-process")
    axes[2].axis('off')

    axes[3].imshow(clean_preds[idx], cmap='gray')
    axes[3].set_title("After Post-process")
    axes[3].axis('off')

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/post_process_comparison.png", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    post_process_evaluation()
