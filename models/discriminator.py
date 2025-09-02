import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchDiscriminatorRad(nn.Module):
    '''PatchGAN discriminator integrating radiomic features, outputs real/fake and view classification '''
    def __init__(self, in_channels=2, ndf=64, rad_dim=102):
        super().__init__()
        self.rad_proj = nn.Linear(rad_dim, 32)

        # Convolutional backbone
        self.model = nn.Sequential(
            nn.Conv2d(in_channels + 32, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # GAN output head (real/fake)
        self.gan_head = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=1)

        # Auxiliary view classifier head (CC vs MLO)
        self.view_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ndf*8, 2)  # 2 classes: CC=0, MLO=1
        )

    def forward(self, x, rad=None):
        if rad is not None:
            proj = self.rad_proj(rad)           
            B, _, H, W = x.shape
            proj = proj.view(B, -1, 1, 1).expand(B, proj.shape[1], H, W)
            x = torch.cat([x, proj], dim=1)

        feat = self.model(x)
        gan_out = self.gan_head(feat)
        view_out = self.view_head(feat)
        return gan_out, view_out
