import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import os

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

def plot_contour_overlay():
    images = np.load("processed_data/images.npy")
    labels = np.load("processed_data/labels.npy")
    labels = np.expand_dims(labels, axis=-1)
    split_idx = int(0.8*len(images))
    x_test = images[split_idx:]
    y_test = labels[split_idx:]

    model = load_model("best_model.h5",
                       custom_objects={"dice_loss":dice_loss,"tversky_loss":tversky_loss})
    y_pred = model.predict(x_test, verbose=0)
    y_pred_bin = (y_pred>0.5).astype(np.float32)

    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.family'] = 'Arial'
    select_num = 3
    fig, axes = plt.subplots(1, select_num, figsize=(15,5))

    for idx in range(select_num):
        img = x_test[idx,:,:,0]
        gt = y_test[idx,:,:,0]
        pred = y_pred_bin[idx,:,:,0]

        axes[idx].imshow(img, cmap="gray")
        # Ground truth green contour
        axes[idx].contour(gt, colors="lime", linewidths=1.2, label="Ground Truth")
        # Prediction red contour
        axes[idx].contour(pred, colors="red", linewidths=1.2, linestyles="--", label="Prediction")
        axes[idx].set_title(f"Sample {idx+1}")
        axes[idx].axis("off")

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/contour_overlay.png", bbox_inches="tight")
    plt.close()
    print(" Contour overlay visualization saved")

if __name__=="__main__":
    plot_contour_overlay()
