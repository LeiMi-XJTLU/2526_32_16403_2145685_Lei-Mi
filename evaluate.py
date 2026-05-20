import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

# Dice loss function (MUST be consistent with training script to load model correctly)
def dice_loss(y_true, y_pred):
    """
    Custom Dice Loss for handling class imbalance in medical image segmentation.
    Args:
        y_true: ground truth label
        y_pred: model prediction
    Returns:
        dice loss value
    """
    smooth = 1e-5
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    loss = 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return loss

# Metric calculation for segmentation evaluation
def calculate_metrics(y_true, y_pred):
    """
    Calculate key evaluation metrics for medical image segmentation.
    Args:
        y_true: Ground truth label
        y_pred: Model prediction
    Returns:
        dice, iou, precision, recall, accuracy
    """
    # Binarize predictions using threshold = 0.5
    y_pred_bin = (y_pred > 0.5).astype(np.float32)

    # Calculate confusion matrix components
    tp = np.sum(y_true * y_pred_bin)  # True Positives
    fp = np.sum((1 - y_true) * y_pred_bin)  # False Positives
    fn = np.sum(y_true * (1 - y_pred_bin))  # False Negatives
    tn = np.sum((1 - y_true) * (1 - y_pred_bin))  # True Negatives

    # Dice Coefficient (F1-Score)
    dice = (2 * tp + 1e-5) / (2 * tp + fp + fn + 1e-5)

    # IoU (Jaccard Index)
    iou = (tp + 1e-5) / (tp + fp + fn + 1e-5)

    # Precision
    precision = (tp + 1e-5) / (tp + fp + 1e-5)

    # Recall (Sensitivity)
    recall = (tp + 1e-5) / (tp + fn + 1e-5)

    # Pixel-wise Accuracy
    accuracy = (tp + tn + 1e-5) / (tp + tn + fp + fn + 1e-5)
    
    return dice, iou, precision, recall, accuracy

# Main evaluation pipeline
def main():
    # Load preprocessed image and label data
    images = np.load("processed_data/images.npy")
    labels = np.load("processed_data/labels.npy")
    labels = np.expand_dims(labels, axis=-1)  # Add channel dimension for labels

    # Split dataset into 80% training and 20% testing
    split_idx = int(0.8 * len(images))
    x_test = images[split_idx:]
    y_test = labels[split_idx:]

    # Load the best-trained model with custom loss function
    model = load_model("best_model.h5", custom_objects={"dice_loss": dice_loss})
    print(" Model loaded successfully!")
    print(f"Test set size: {len(x_test)} samples\n")

    # Generate predictions on test set
    y_pred = model.predict(x_test, verbose=1)

    # Initialize lists to store evaluation results
    dice_list = []
    iou_list = []
    precision_list = []
    recall_list = []
    acc_list = []

    # Compute metrics for each test sample
    for i in range(len(x_test)):
        dice, iou, precision, recall, acc = calculate_metrics(y_test[i], y_pred[i])
        dice_list.append(dice)
        iou_list.append(iou)
        precision_list.append(precision)
        recall_list.append(recall)
        acc_list.append(acc)

    # Print final average evaluation results
    print("=" * 60)
    print("          TEST SET EVALUATION RESULTS          ")
    print("=" * 60)
    print(f"Mean Dice Coefficient: {np.mean(dice_list):.4f}")
    print(f"Mean IoU:              {np.mean(iou_list):.4f}")
    print(f"Mean Precision:        {np.mean(precision_list):.4f}")
    print(f"Mean Recall:           {np.mean(recall_list):.4f}")
    print(f"Mean Accuracy:         {np.mean(acc_list):.4f}")
    print("=" * 60)
    print("\n Evaluation completed successfully!")

if __name__ == "__main__":
    main()
