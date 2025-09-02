import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to your results file
results_file = "path/to/results.csv"   # <-- change this

# Load CSV
df = pd.read_csv(results_file)

# ---- Plot losses ----
plt.figure(figsize=(10,5))
plt.plot(df["epoch"], df["loss_G"], label="Generator Loss", color="blue")
plt.plot(df["epoch"], df["loss_D"], label="Discriminator Loss", color="red")
plt.plot(df["epoch"], df["loss_GAN"], label="GAN Loss", linestyle="--", color="purple")
plt.plot(df["epoch"], df["loss_L1"], label="L1 Loss", linestyle="--", color="green")
plt.plot(df["epoch"], df["loss_percep"], label="Perceptual Loss", linestyle="--", color="orange")
plt.plot(df["epoch"], df["loss_rad"], label="Radiomics Loss", linestyle="--", color="brown")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Losses")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(results_file), "loss_curves.png"))
plt.close()

# ---- Plot validation metrics ----
plt.figure(figsize=(10,5))
plt.plot(df["epoch"], df["val_L1"], label="Validation L1", color="green")
plt.plot(df["epoch"], df["val_SSIM"], label="Validation SSIM", color="blue")
plt.plot(df["epoch"], df["val_PSNR"], label="Validation PSNR", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.title("Validation Metrics")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(results_file), "val_metrics.png"))
plt.close()

print("✅ Saved loss_curves.png and val_metrics.png")
