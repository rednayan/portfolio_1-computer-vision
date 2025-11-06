import matplotlib.pyplot as plt
import numpy as np
import pickle
import json
import os

TRAINING_HISTORY_FILE = 'training_history.pkl'
METRICS_OUTPUT_FILE = 'evaluation_metrics.json'
OUTPUT_PLOTS_DIR = 'plots'


def load_history(file_path: str) -> dict:
    """Loads the training history (losses, LR) from the pickle file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training history not found: {file_path}. Run model_trainer.py first.")
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_metrics(file_path: str) -> dict:
    """Loads the final evaluation metrics from the JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Evaluation metrics not found: {file_path}. Run evaluation_metrics.py first.")
    with open(file_path, 'r') as f:
        return json.load(f)


def plot_loss_curves(history: dict, output_dir: str):
    """Plots the training and validation loss over epochs."""
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Total Loss)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'))
    plt.close()
    print("Saved loss curves plot.")

def plot_learning_rate(history: dict, output_dir: str):
    """Plots the learning rate schedule over epochs."""
    epochs = range(1, len(history['lr']) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['lr'], 'g', label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'learning_rate.png'))
    plt.close()
    print("Saved learning rate plot.")

def print_final_metrics(metrics: dict):
    """Prints a clean summary of the final evaluation metrics."""
    print("\n--- Final Evaluation Metrics Summary ---")

    print(f"mAP (IoU=0.50:0.95): {metrics.get('map', 'N/A'):.4f}")
    print(f"mAP@50 (IoU=0.50):    {metrics.get('map_50', 'N/A'):.4f}")
    print(f"mAP@75 (IoU=0.75):    {metrics.get('map_75', 'N/A'):.4f}")
    print(f"Recall (Max Dets=100): {metrics.get('mar_100', 'N/A'):.4f}")
    print("-" * 40)

if __name__ == '__main__':
    try:
        os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

        history = load_history(TRAINING_HISTORY_FILE)
        metrics = load_metrics(METRICS_OUTPUT_FILE)

        plot_loss_curves(history, OUTPUT_PLOTS_DIR)
        plot_learning_rate(history, OUTPUT_PLOTS_DIR)

        print_final_metrics(metrics)

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")