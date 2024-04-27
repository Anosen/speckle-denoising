import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.unets.batchrenorm import BatchRenorm2d

class Block_D_Unet(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Block_D_Unet, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(0.1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        return x

class Block1(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Block1, self).__init__()
        self.conv_transpose = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation =2, stride = stride)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.05)
        self.batch_renorm = BatchRenorm2d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.batch_renorm(x)
        return x

class Block2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block2, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=2,dilation=2, output_padding=0)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(0.05)
        self.batch_renorm = BatchRenorm2d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.batch_renorm(x)
        return x
