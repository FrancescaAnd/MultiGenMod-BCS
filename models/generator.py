import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

class UNetBlock(nn.Module):
    '''Basic UNet block for downsampling or upsampling with optional batch norm and dropout '''
    def __init__(self, in_channels, out_channels, down=True, use_in=True, activation='relu', dropout=False):
        super().__init__()
        layers = []
        if down:
            layers.append(nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False))
            if use_in: layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True) if activation=='leaky' else nn.ReLU(inplace=True))
        else:
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False))
            if use_in: layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.1))
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return self.block(x)


class RadiomicsFiLM(nn.Module):
    '''Given radiomics vector r (B, F), produce per-layer gamma/beta parameters'''

    def __init__(self, in_dim, film_channels):
        # film_channels: list of ints, for each modulation layer produce gamma/beta of that length
        super().__init__()
        self.in_dim = in_dim
        self.film_channels = film_channels
        # small MLP: project radiomics to a shared hidden space
        hidden = 512
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        # create per-layer heads
        self.heads = nn.ModuleList()
        for c in film_channels:
            # output gamma and beta per channel (2*c)
            self.heads.append(nn.Linear(hidden, 2 * c))

    def forward(self, r):
        h = self.mlp(r)  
        outs = []
        for head in self.heads:
            out = head(h)  
            outs.append(out)
        # outs: list of tensors per layer
        return outs


class UNetGeneratorRad(nn.Module):
    '''UNet generator integrating radiomic FiLM conditioning'''

    def __init__(self, in_channels=2, out_channels=2, rad_dim=102):
        super().__init__()
        # Encoder
        self.enc1 = UNetBlock(in_channels, 64, down=True, use_in=False, activation='leaky')   
        self.enc2 = UNetBlock(64, 128, down=True, use_in=True, activation='leaky')           
        self.enc3 = UNetBlock(128, 256, down=True, use_in=True, activation='leaky')          
        self.enc4 = UNetBlock(256, 512, down=True, use_in=True, activation='leaky')          
        self.enc5 = UNetBlock(512, 512, down=True, use_in=True, activation='leaky')          
        self.enc6 = UNetBlock(512, 512, down=True, use_in=True, activation='leaky')          
        self.enc7 = UNetBlock(512, 512, down=True, use_in=True, activation='leaky')          

        # Decoder
        self.dec7 = UNetBlock(512, 512, down=False, use_in=True)      
        self.dec6 = UNetBlock(1024, 512, down=False, dropout=True, use_in=True)     
        self.dec5 = UNetBlock(1024, 512, down=False, use_in=True)     
        self.dec4 = UNetBlock(1024, 256, down=False, use_in=True)     
        self.dec3 = UNetBlock(512, 128, down=False, use_in=True)      
        self.dec2 = UNetBlock(256, 64, down=False, use_in=True)       
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

        # FiLM conditioning: produce gamma/beta for decoder layers where we modulate features
        film_channels = [512, 512, 512, 256, 128, 64]  
        self.rad_film = RadiomicsFiLM(rad_dim, film_channels)

    def _apply_film(self, feat, film_params):
        B, C, H, W = feat.shape
        gamma_beta = film_params.view(B, 2, C)  
        gamma = gamma_beta[:,0].view(B, C, 1, 1)
        beta  = gamma_beta[:,1].view(B, C, 1, 1)
        # (1 + gamma) * feat + beta 
        return feat * (1.0 + gamma) + beta

    def forward(self, x, rad=None):
        
        # Encoder with checkpointing
        e1 = self.enc1(x)
        e2 = cp.checkpoint(self.enc2, e1, use_reentrant=False)
        e3 = cp.checkpoint(self.enc3, e2, use_reentrant=False)
        e4 = cp.checkpoint(self.enc4, e3, use_reentrant=False)
        e5 = cp.checkpoint(self.enc5, e4, use_reentrant=False)
        e6 = cp.checkpoint(self.enc6, e5, use_reentrant=False)
        e7 = cp.checkpoint(self.enc7, e6, use_reentrant=False)

        # Compute FiLM params 
        if rad is not None:
            film_outs = self.rad_film(rad)  
        else:
            film_outs = [None] * 6

        # Decoder with checkpointing and FiLM
        d7 = cp.checkpoint(self.dec7, e7, use_reentrant=False)
        if film_outs[0] is not None:
            d7 = self._apply_film(d7, film_outs[0])

        d7_up = F.interpolate(d7, size=e6.shape[2:], mode='bilinear', align_corners=False)
        d6 = cp.checkpoint(self.dec6, torch.cat([d7_up, e6], dim=1), use_reentrant=False)
        if film_outs[1] is not None:
            d6 = self._apply_film(d6, film_outs[1])

        d6_up = F.interpolate(d6, size=e5.shape[2:], mode='bilinear', align_corners=False)
        d5 = cp.checkpoint(self.dec5, torch.cat([d6_up, e5], dim=1), use_reentrant=False)
        if film_outs[2] is not None:
            d5 = self._apply_film(d5, film_outs[2])

        d5_up = F.interpolate(d5, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d4 = cp.checkpoint(self.dec4, torch.cat([d5_up, e4], dim=1), use_reentrant=False)
        if film_outs[3] is not None:
            d4 = self._apply_film(d4, film_outs[3])

        d4_up = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = cp.checkpoint(self.dec3, torch.cat([d4_up, e3], dim=1), use_reentrant=False)
        if film_outs[4] is not None:
            d3 = self._apply_film(d3, film_outs[4])

        d3_up = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = cp.checkpoint(self.dec2, torch.cat([d3_up, e2], dim=1), use_reentrant=False)
        if film_outs[5] is not None:
            d2 = self._apply_film(d2, film_outs[5])

        d2_up = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d2_up, e1], dim=1))

        return d1