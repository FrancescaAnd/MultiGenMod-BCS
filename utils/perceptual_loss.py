import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

class VGGPerceptual(nn.Module):
    ''' feature-extraction network based on the pretrained VGG16 model for computing perceptual losses'''

    def __init__(self, layers=['3','8','15','22'], device='cuda'):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.vgg = vgg
        self.layers = layers
        for param in self.vgg.parameters():
            param.requires_grad = False  # freeze
    def forward(self, x):
        feats = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if str(i) in self.layers:
                feats.append(x)
        return feats

def perceptual_loss(fake, real, net):
    f_fake = net(fake)
    f_real = net(real)
    loss = 0
    for ff, fr in zip(f_fake, f_real):
        loss += nn.functional.l1_loss(ff, fr)
    return loss
