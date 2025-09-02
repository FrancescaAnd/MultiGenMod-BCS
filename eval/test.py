import argparse
from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import DataLoader
from dataset.dataset import MammogramFullDataset
from dataset.radiomics_loader import RadiomicsLoader
from models.generator import UNetGeneratorRad
from eval import eval
from torchvision import transforms
from tqdm import tqdm
import time

# ---------- CLI Arguments ---------- #
parser = argparse.ArgumentParser(description="Test Generator on Mammogram Dataset")
parser.add_argument("--dataset", type=str, choices=["cbis", "inbreast"], required=True)
parser.add_argument("--ckpt_path", type=str, required=True)
parser.add_argument("--roi_path", type=str, required=True)
parser.add_argument("--radiomics_csv", type=str, required=True)
parser.add_argument("--save_dir", type=str, default="generated_test_images")
parser.add_argument("--batch_size", type=int, default=4)
args = parser.parse_args()

# ---------- Device ---------- #
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Transform ---------- #
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Normalize([0.5], [0.5])
])

# ---------- Load test dataset ---------- #
df_roi = pd.read_pickle(Path(args.roi_path))
rad_loader = RadiomicsLoader(Path(args.radiomics_csv))
test_dataset = MammogramFullDataset(df_roi, rad_loader, patient_ids=df_roi["patient_id"].unique(), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# ---------- Load generator ---------- #
rad_dim = 102
G = UNetGeneratorRad(in_channels=2, out_channels=2, rad_dim=rad_dim).to(device)
checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=True)
G.load_state_dict(checkpoint['G_state_dict'])

# ---------- Evaluate with progress bar and timing ---------- #
start_time = time.time()
print("Evaluating test set...")
metrics = eval(G, test_loader, device, epoch="test", save_dir=args.save_dir, use_tqdm=True)
elapsed_time = time.time() - start_time

print(f"\nTest completed in {elapsed_time:.1f} seconds")
print(f"Test L1: {metrics['L1']:.4f}, SSIM: {metrics['SSIM']:.4f}, PSNR: {metrics['PSNR']:.2f}")
