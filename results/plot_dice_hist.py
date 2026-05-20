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

def cal_single_dice(y_true, y_pred):
    smooth=1e-5
    y_pred_bin = (y_pred>0.5).astype(np.float32)
    tp = np.sum(y_true*y_pred_bin)
    return (2*tp+smooth)/(np.sum(y_true)+np.sum(y_pred_bin)+smooth)

def plot_dice_distribution():
    images = np.load("processed_data/images.npy")
    labels = np.load("processed_data/labels.npy")
    labels = np.expand_dims(labels, axis=-1)
    split_idx = int(0.8*len(images))
    x_test = images[split_idx:]
    y_test = labels[split_idx:]

    model = load_model("best_model.h5",
                       custom_objects={"dice_loss":dice_loss,"tversky_loss":tversky_loss})
    y_pred = model.predict(x_test, verbose=0)

    dice_list = []
    for i in range(len(x_test)):
        d = cal_single_dice(y_test[i], y_pred[i])
        dice_list.append(d)

    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.family'] = 'Arial'
    plt.figure(figsize=(7,5))
    plt.hist(dice_list, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(dice_list), color='red', linestyle='--',
                label=f'Mean Dice = {np.mean(dice_list):.4f}')
    plt.xlabel('Dice Coefficient')
    plt.ylabel('Sample Number')
    plt.title('Distribution of Dice Coefficient on Test Set')
    plt.legend()
    plt.grid(alpha=0.3)
    os.makedirs("results", exist_ok=True)
    plt.tight_layout()
    plt.savefig("results/dice_hist.png", bbox_inches="tight")
    plt.close()
    print(" Dice distribution histogram saved")

if __name__=="__main__":
    plot_dice_distribution()
