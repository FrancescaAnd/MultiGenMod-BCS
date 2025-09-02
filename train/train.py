import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.generator import UNetGeneratorRad
from models.discriminator import PatchDiscriminatorRad
from models.rad_predictor import RadPredictor
from dataset.radiomics_loader import RadiomicsLoader
from utils.perceptual_loss import VGGPerceptual, perceptual_loss
import pandas as pd
import os, time, argparse,re, csv
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import trange
from pathlib import Path
from pytorch_msssim import ssim
from dataset.dataset import MammogramFullDataset, Augmentation
from sklearn.model_selection import train_test_split
from eval.eval import eval


# ----------Modules---------------#
def add_input_noise(x, std=0.03):
    return x + torch.randn_like(x) * std

def linear_ramp(epoch, start, end, ramp_epochs):
    t = min(1.0, epoch / float(max(1, ramp_epochs)))
    return start + (end - start) * t

def detect_collapse(tensor, std_thresh=1e-4):
    tstd = float(tensor.detach().cpu().std().item())
    return (tstd < std_thresh) or torch.isnan(tensor).any() or torch.isinf(tensor).any()


#--------------Main-----------#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script with dataset switch")
    parser.add_argument("--dataset", type=str, choices=["cbis", "inbreast"], required=True, help="Dataset name")
    args = parser.parse_args()

    configs = {
        "cbis": {
            "roi_df": Path("data/CBIS-DDSM/pkl/two_views_roi.pkl"),
            "radiomics": Path("data/cbis_radiomics.csv"),
        },
        "inbreast": {
            "roi_df": Path("data/INbreast/pkl/two_views_roi.pkl"),
            "radiomics": Path("data/inbreast_radiomics.csv"),
        },
    }

    cfg = configs[args.dataset]

    # Load radiomics and dataframe
    df_roi = pd.read_pickle(cfg["roi_df"])
    rad_loader = RadiomicsLoader(cfg["radiomics"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data augmentation
    augment = Augmentation(max_rotation=5, hflip_prob=0.5)

    # Resize and normalize
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Normalize([0.5], [0.5])
    ])
    batch_size = 4

    patient_labels = df_roi.groupby("patient_id")["lesion"].agg(lambda x: x.mode()[0]).to_dict()  # assign a label per patient for stratification
    df_roi["patient_label"] = df_roi["patient_id"].map(patient_labels)
    patients = df_roi["patient_id"].unique()
    labels = [patient_labels[p] for p in patients]

    train_patients, temp_patients = train_test_split(
        patients, test_size=0.3, random_state=42, shuffle=True
    )
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=0.5, random_state=42, shuffle=True
    )

    # Datasets
    train_dataset = MammogramFullDataset(df_roi, rad_loader, patient_ids=train_patients, augment=None, transform=transform)
    val_dataset   = MammogramFullDataset(df_roi, rad_loader, patient_ids=val_patients, transform=transform)
    test_dataset  = MammogramFullDataset(df_roi, rad_loader, patient_ids=test_patients, transform=transform)

    print(f"Train: {len(train_dataset)} patients | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_dataset)} patients | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
  
    # Models
    rad_dim = 102
    G = UNetGeneratorRad(in_channels=2, out_channels=2, rad_dim=rad_dim).to(device)
    D = PatchDiscriminatorRad(in_channels=4, ndf=64, rad_dim=rad_dim).to(device)
    rad_pred = RadPredictor(in_channels=1, out_dim=rad_dim).to(device)
    percep_net = VGGPerceptual().to(device)

    # Losses
    criterion_GAN = nn.MSELoss()
    criterion_L1  = nn.L1Loss()
    criterion_id  = nn.L1Loss()
    criterion_rad = nn.L1Loss()

    # Optimizers
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5,0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5,0.999))
    opt_rad = torch.optim.Adam(rad_pred.parameters(), lr=1e-4)

    scaler = torch.amp.GradScaler()
    real_val, fake_val = 0.9, 0.1

    # Early stopping parameters 
    best_ssim = 0.0        
    best_epoch = 0          
    patience = 10          

    # Best ckpt path
    best_val_l1 = float('inf')
    best_ckpt_path = None

    save_dir = "generated_images_full"
    os.makedirs(save_dir, exist_ok=True)

    # Checkpoint directories
    ckpt_train_dir = os.path.join("checkpoints/train/", args.dataset)
    os.makedirs(ckpt_train_dir, exist_ok=True)
    print(f"Using checkpoint directory: {ckpt_train_dir}")
    ckpt_val_dir = os.path.join("checkpoints/val/", args.dataset)
    os.makedirs(ckpt_val_dir, exist_ok=True)


    num_epochs = 50
    accum_steps = 4  

    # Resume from latest checkpoint
    latest_ckpt = None
    ckpt_files = [f for f in os.listdir(ckpt_train_dir) if f.endswith(".pt")]
    if ckpt_files:
        def epoch_from_name(name):
            m = re.search(r"(\d+)(?=\.pt$)", name)
            return int(m.group(1)) if m else -1
        ckpt_files_sorted = sorted(ckpt_files, key=epoch_from_name)
        latest_ckpt = os.path.join(ckpt_train_dir, ckpt_files_sorted[-1])
    else:
        latest_ckpt = None

    start_epoch = 0
    if latest_ckpt is not None:
        checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=True)
        G.load_state_dict(checkpoint['G_state_dict'])
        D.load_state_dict(checkpoint['D_state_dict'])
        opt_G.load_state_dict(checkpoint['opt_G_state_dict'])
        opt_D.load_state_dict(checkpoint['opt_D_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {start_epoch}")

    d_steps = 1 
    grad_clip = 1.0  # clip norm

    # -------Saving results in a csv file -----#
    results_file = os.path.join(ckpt_train_dir, "results.csv")
    if not os.path.exists(results_file):
        with open(results_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "loss_G", "loss_D",
                "loss_GAN", "loss_L1", "loss_percep", "loss_rad",
                "val_L1", "val_SSIM", "val_PSNR"
            ])

    # --------------- Training loop -------------#
    
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()

        #Ramp lambdas slowly (TEST 1)
        lambda_L1     = linear_ramp(epoch, 5.0, 1.0, ramp_epochs=30)    
        lambda_adv    = linear_ramp(epoch, 0.1, 1.0, ramp_epochs=30)    
        lambda_percep = linear_ramp(epoch, 0.5, 1.0, ramp_epochs=30)
        lambda_rad    = linear_ramp(epoch, 0.0, 1.0, ramp_epochs=60)
        lambda_ssim   = 0.2
        
        #(TEST 2)
        # lambda_L1     = linear_ramp(epoch, 5.0, 2.0, ramp_epochs=50)  
        # lambda_adv    = linear_ramp(epoch, 0.05, 0.5, ramp_epochs=50) 
        # lambda_percep = linear_ramp(epoch, 0.5, 1.5, ramp_epochs=30)   
        # lambda_rad    = linear_ramp(epoch, 0.0, 1.0, ramp_epochs=60)
        # lambda_ssim   = 0.5   # stronger SSIM to preserve structure

        loop = trange(len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for i, batch in enumerate(train_loader):

            # Move inputs
            cc_input = batch['cc_input'].to(device)
            mlo_input = batch['mlo_input'].to(device)
            rad = batch['radiomics'].to(device)

            # --------------Generator------------- #
            opt_G.zero_grad()
            with torch.amp.autocast('cuda'):
                fake_mlo = G(cc_input, rad)
                if fake_mlo.shape[1] != 2:
                    fake_mlo = fake_mlo[:, :2, :, :]

                disc_input_fake = torch.cat([cc_input, fake_mlo], dim=1)
                pred_fake, view_fake = D(disc_input_fake, rad)

                tgt_real = torch.full_like(pred_fake, real_val, device=device)
                loss_GAN = criterion_GAN(pred_fake, tgt_real) * lambda_adv
                target_view = torch.ones(cc_input.size(0), dtype=torch.long, device=device)
                loss_view = F.cross_entropy(view_fake, target_view)
                loss_L1 = criterion_L1(fake_mlo[:,0:1], mlo_input[:,0:1]) * lambda_L1

                fake_vgg = F.interpolate(fake_mlo[:,0:1], size=(224,224), mode='bilinear', align_corners=False).repeat(1,3,1,1).float()
                real_vgg = F.interpolate(mlo_input[:,0:1], size=(224,224), mode='bilinear', align_corners=False).repeat(1,3,1,1).float()
                with torch.amp.autocast('cuda', enabled=False):
                    loss_percep = perceptual_loss(fake_vgg, real_vgg, percep_net) * lambda_percep

                loss_ssim = (1 - ssim(fake_mlo[:,0:1], mlo_input[:,0:1], data_range=1.0, size_average=True)) * lambda_ssim
                pred_rad = rad_pred(fake_mlo[:,0:1])
                loss_rad = criterion_rad(pred_rad, rad) * lambda_rad

                loss_G = loss_GAN + loss_view + loss_L1 + loss_percep + loss_ssim + loss_rad 

            scaler.scale(loss_G).backward()

            # step generator with gradient clipping
            if (i+1) % accum_steps == 0 or (i+1) == len(train_loader):
                # unscale before clipping
                scaler.unscale_(opt_G)
                torch.nn.utils.clip_grad_norm_(G.parameters(), grad_clip)
                scaler.step(opt_G)
                scaler.update()
                opt_G.zero_grad()

            # ---------- Discriminator --------#
            if (i % d_steps) == 0: # update discriminator every d_steps
                opt_D.zero_grad()
                with torch.amp.autocast('cuda'):
                    real_pair = torch.cat([cc_input, mlo_input], dim=1)
                    real_in = add_input_noise(real_pair, std=0.03)
                    fake_in = add_input_noise(disc_input_fake.detach(), std=0.03)
                    pred_real, view_real = D(real_in, rad)
                    pred_fake_detach, view_fake_detach = D(fake_in, rad)
                    tgt_real = torch.full_like(pred_real, real_val, device=device)
                    tgt_fake = torch.full_like(pred_fake_detach, fake_val, device=device)
                    loss_D_gan = 0.5 * (criterion_GAN(pred_real, tgt_real) + criterion_GAN(pred_fake_detach, tgt_fake)) / accum_steps

                    target_view_real_mlo = torch.ones(cc_input.size(0), dtype=torch.long, device=device)
                    target_view_fake_mlo = torch.ones(cc_input.size(0), dtype=torch.long, device=device)
                    cc_pair = torch.cat([cc_input, cc_input], dim=1)
                    cc_pair = add_input_noise(cc_pair, std=0.03)
                    _, view_cc = D(cc_pair, rad)
                    target_view_cc = torch.zeros(cc_input.size(0), dtype=torch.long, device=device)
                    loss_D_view = (F.cross_entropy(view_real, target_view_real_mlo) +
                                F.cross_entropy(view_fake_detach, target_view_fake_mlo) +
                                F.cross_entropy(view_cc, target_view_cc)) / (3 * accum_steps)
                    loss_D = loss_D_gan + loss_D_view

                scaler.scale(loss_D).backward()
                if (i+1) % accum_steps == 0 or (i+1) == len(train_loader):
                    scaler.unscale_(opt_D)
                    torch.nn.utils.clip_grad_norm_(D.parameters(), grad_clip)
                    scaler.step(opt_D)
                    scaler.update()
                    opt_D.zero_grad()

                # Update tqdm bar 
                loop.set_postfix({
                    'Loss_G': f"{loss_G.item():.3f}",
                    'GAN': f"{loss_GAN.item():.3f}",
                    'L1': f"{loss_L1.item():.3f}",
                    'Percep': f"{loss_percep.item():.3f}",
                    'Rad': f"{loss_rad.item():.3f}",
                    'Loss_D': f"{loss_D.item():.3f}",
                    'Time': f"{(time.time() - start_time):.1f}s"
                })
                loop.update(1)
            torch.cuda.empty_cache()
        

        # ---------- Save checkpoint --------#
        ckpt_path = os.path.join(ckpt_train_dir, f"gan_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'G_state_dict': G.state_dict(),
            'D_state_dict': D.state_dict(),
            'opt_G_state_dict': opt_G.state_dict(),
            'opt_D_state_dict': opt_D.state_dict(),
            'scaler_state_dict': scaler.state_dict()
        }, ckpt_path)

        # ---------- Evaluation --------#
        val_metrics = eval(G, val_loader, device, epoch, save_dir=ckpt_val_dir)
        print(f"Epoch {epoch+1} | Validation L1: {val_metrics['L1']:.4f}")
        val_ssim = val_metrics.get('SSIM', 0.0)

        val_psnr = val_metrics.get('PSNR', 0.0)

        with open(results_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                float(loss_G.item()), float(loss_D.item()),
                float(loss_GAN.item()), float(loss_L1.item()), float(loss_percep.item()), float(loss_rad.item()),
                float(val_metrics['L1']), float(val_ssim), float(val_psnr)
            ])
 

        # Early stopping
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            best_epoch = epoch
            torch.save({
                'epoch': epoch + 1,
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'opt_G_state_dict': opt_G.state_dict(),
                'opt_D_state_dict': opt_D.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }, os.path.join(ckpt_train_dir, 'best_model.pt'))
        else:
            if epoch - best_epoch >= patience:
                print(f"No improvement for {patience} epochs. Early stopping at epoch {epoch+1}.")
                break

        torch.cuda.empty_cache()

        loop.close()
        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{num_epochs} "
            f"| Time: {epoch_time:.1f}s "
            f"| G: {loss_G.item():.3f} "
            f"(GAN {loss_GAN.item():.3f}, L1 {loss_L1.item():.3f}, Percep {loss_percep.item():.3f}, Rad {loss_rad.item():.3f}) "
            f"| D: {loss_D.item():.3f}"
        )

        # ---------- Save generated images --------#
        batch = next(iter(train_loader))
        cc_input = batch['cc_input'].to(device)
        mlo_input = batch['mlo_input'].to(device)
        rad = batch['radiomics'].to(device)

        with torch.no_grad():
            fake_mlo = G(cc_input, rad)

        cc_img = cc_input[:,0:1].cpu()
        mlo_img = mlo_input[:,0:1].cpu()
        fake_mlo = fake_mlo.cpu()

        fig, axs = plt.subplots(1, 3, figsize=(12,4))
        axs[0].imshow(cc_img[0,0], cmap='gray'); axs[0].set_title("CC Real")
        axs[1].imshow(mlo_img[0,0], cmap='gray'); axs[1].set_title("MLO Real")
        axs[2].imshow(fake_mlo[0,0], cmap='gray'); axs[2].set_title("MLO Fake")

        for ax in axs:
            ax.axis('off')

        save_path = os.path.join(save_dir, f"epoch_{epoch+1}.png")
        plt.savefig(save_path)
        plt.close(fig)

        print(f"Saved generated images to {save_path}")

        