import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def plot_training_curve():
    # Load saved training history
    with open("training_history.pkl", "rb") as f:
        history = pickle.load(f)

    epochs = range(1, len(history['loss']) + 1)

    # Set high-quality figure style
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.family'] = 'Arial'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot training & validation loss
    ax1.plot(epochs, history['loss'], 'b-', linewidth=1.5, label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', linewidth=1.5, label='Validation Loss')
    ax1.set_title('Training and Validation Loss', fontsize=12)
    ax1.set_xlabel('Epochs', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Plot training & validation accuracy
    ax2.plot(epochs, history['accuracy'], 'b-', linewidth=1.5, label='Training Accuracy')
    ax2.plot(epochs, history['val_accuracy'], 'r-', linewidth=1.5, label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy', fontsize=12)
    ax2.set_xlabel('Epochs', fontsize=10)
    ax2.set_ylabel('Accuracy', fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/training_curve.png", bbox_inches="tight")
    plt.close()

    print(" Training curve saved to results/training_curve.png")

if __name__ == "__main__":
    plot_training_curve()
