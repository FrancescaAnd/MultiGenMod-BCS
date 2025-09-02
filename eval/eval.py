import os
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def denormalize(tensor):
    '''Convert from [-1,1] to [0,1].'''
    return (tensor + 1.0) / 2.0

def compute_ssim_batch(fake, real):
    '''Compute mean SSIM over batch '''
    ssim_total = 0.0
    B = fake.size(0)
    for i in range(B):
        f = fake[i,0].cpu().numpy()
        r = real[i,0].cpu().numpy()
        ssim_total += ssim(r, f, data_range=1.0)
    return ssim_total / B

def compute_psnr_batch(fake, real):
    '''Compute mean PSNR over batch '''
    psnr_total = 0.0
    B = fake.size(0)
    for i in range(B):
        f = fake[i,0].cpu().numpy()
        r = real[i,0].cpu().numpy()
        psnr_total += psnr(r, f, data_range=1.0)
    return psnr_total / B

def eval(G, loader, device, epoch, save_dir):
    '''Evaluate the generator on a loader and save sample images, 
    Returns L1, SSIM, PSNR'''
    
    G.eval()
    os.makedirs(os.path.join(save_dir, f"epoch_{epoch}"), exist_ok=True)

    total_l1 = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    n_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            cc_input = batch['cc_input'].to(device)
            mlo_input = batch['mlo_input'].to(device)
            rad = batch['radiomics'].to(device)
            fake_mlo = G(cc_input, rad)
            
            # Denormalize to [0,1] for metrics
            fake_mlo_d = denormalize(fake_mlo[:,0:1])
            mlo_input_d = denormalize(mlo_input[:,0:1])

            # Compute metrics
            total_l1 += F.l1_loss(fake_mlo[:,0:1], mlo_input[:,0:1], reduction='sum').item()
            total_ssim += compute_ssim_batch(fake_mlo_d, mlo_input_d)
            total_psnr += compute_psnr_batch(fake_mlo_d, mlo_input_d)
            n_batches += 1

            # Save first N images for visualization
            N = min(4, cc_input.size(0))
            for j in range(N):
                cc_img = denormalize(cc_input[j].cpu())[0:1]
                fake_img = fake_mlo_d[j].cpu()
                real_img = mlo_input_d[j].cpu()

                vis_img = torch.cat([cc_img, fake_img, real_img], dim=2)  # width concat
                save_path = os.path.join(save_dir, f"epoch_{epoch}", f"sample_{i*N+j}.png")
                vutils.save_image(vis_img, save_path, normalize=True)

    N_total = len(loader.dataset)
    avg_l1 = total_l1 / N_total
    avg_ssim = total_ssim / n_batches
    avg_psnr = total_psnr / n_batches


    print(f"L1: {avg_l1:.4f}")
    print(f"SSIM: {avg_ssim:.4f}")
    print(f"PSNR: {avg_psnr:.2f}")

    return {'L1': avg_l1, 'SSIM': avg_ssim, 'PSNR': avg_psnr}
