import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_training_metrics(dataset: str, save_dir: str = "plots"):
 
    # Map dataset to CSV path
    csv_paths = {
        "inbreast": "results/results_inbreast.csv",
        "cbis": "results/results_inbreast.csv"
    }

    if dataset not in csv_paths:
        raise ValueError(f"Dataset '{dataset}' not recognized. Choose from {list(csv_paths.keys())}.")

    csv_path = Path(csv_paths[dataset])
    df = pd.read_csv(csv_path)

    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Plot L1
    plt.figure(figsize=(8,5))
    plt.plot(df['epoch'], df['val_L1'], label='Validation L1', color='tab:blue')
    plt.xlabel('Epoch')
    plt.ylabel('L1 Loss')
    plt.title(f'{dataset.upper()} Validation L1 over Epochs')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(save_dir) / f"{dataset}_val_L1.png")
    plt.show()

    # Plot SSIM
    plt.figure(figsize=(8,5))
    plt.plot(df['epoch'], df['val_SSIM'], label='Validation SSIM', color='tab:green')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title(f'{dataset.upper()} Validation SSIM over Epochs')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(save_dir) / f"{dataset}_val_SSIM.png")
    plt.show()

    # Plot PSNR
    plt.figure(figsize=(8,5))
    plt.plot(df['epoch'], df['val_PSNR'], label='Validation PSNR', color='tab:red')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title(f'{dataset.upper()} Validation PSNR over Epochs')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(save_dir) / f"{dataset}_val_PSNR.png")
    plt.show()
