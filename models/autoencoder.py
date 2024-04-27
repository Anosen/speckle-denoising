import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.base import BaseModel

# Define the Dilated Convolutional Autoencoder
class DilatedConvAutoencoder(BaseModel):
    def __init__(self):
        super(DilatedConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.1),

            nn.MaxPool2d(2, 2),

            nn.Dropout(0.2),

            nn.BatchNorm2d(256),

            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.1),

            nn.Dropout(0.2),
        )

        # Decoder that mirrors the encoder's dilation pattern
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.1),

            # Upsampling
            # nn.Upsample(scale_factor=2, mode='nearest'),  # Upsampling to 128x128
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  # Replaces nn.Upsample

            nn.Dropout(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.1),

            nn.Dropout(0.2),

            nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 1, kernel_size=3, padding=2, dilation=2),
            nn.Tanh()  # Normalize output pixels
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x