import torch.nn as nn

class RadPredictor(nn.Module):
    '''Small radiomics predictor (learns to predict rad vector from generated image features)'''

    def __init__(self, in_channels=1, out_dim=102):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),  
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        return self.fc(h)  
