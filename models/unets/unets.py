import torch
from torch import nn
from torch.nn import functional as F

from models.base import BaseModel
from models.unets.blocks import Block_D_Unet, Block1, Block2

class DenoisingUNet(BaseModel):
    def __init__(self):
        super(DenoisingUNet, self).__init__()

        self.block1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(0.1)
                                    )
        self.block2 = nn.Sequential(nn.MaxPool2d(2, 2),
                                    Block_D_Unet(32, 64)
                                    )
        self.block3 = nn.Sequential(nn.MaxPool2d(2, 2),
                                    Block_D_Unet(64, 128)
                                    )
        self.block4 = nn.Sequential(nn.MaxPool2d(2, 2),
                                    Block_D_Unet(128, 256)
                                    )
        self.bottom = nn.Sequential(nn.MaxPool2d(2, 2),
                                     Block_D_Unet(256, 512),
                                     nn.ConvTranspose2d(512, 512, 2, output_padding=0)
                                     )

        # Decoder
        self.dec_block_1 = Block_D_Unet(768, 256)
        self.dec_block_2 = nn.Sequential(Block_D_Unet(384, 128),
                                          nn.BatchNorm2d(128))
        self.dec_block_3 = nn.Sequential(Block_D_Unet(192, 64),
                                          nn.BatchNorm2d(64))
        self.dec_block_4 = nn.Sequential(Block_D_Unet(96, 32),
                                          nn.BatchNorm2d(32))
        self.last_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.bottom(x4)
        x5_interpol = F.interpolate(x5, size=x4.shape[2:], mode='bilinear', align_corners=False)
        cat_1 = torch.cat([x5_interpol, x4], dim=1)
        x6 = self.dec_block_1(cat_1)
        x6_interpol = F.interpolate(x6, size=x3.shape[2:], mode='bilinear', align_corners=False)
        cat_2 = torch.cat([x6_interpol, x3], dim=1)
        x7 = self.dec_block_2(cat_2)
        x7_interpol = F.interpolate(x7, size=x2.shape[2:], mode='bilinear', align_corners=False)
        cat_3 = torch.cat([x7_interpol, x2], dim=1)
        x8 = self.dec_block_3(cat_3)
        x8_interpol = F.interpolate(x8, size=x1.shape[2:], mode='bilinear', align_corners=False)
        cat_4 = torch.cat([x8_interpol, x1], dim=1)
        x9 = self.dec_block_4(cat_4)
        x10 = self.last_conv(x9)
        return x10

class BatchRenormalizationUNet(BaseModel):
    def __init__(self):
        super(BatchRenormalizationUNet, self).__init__()

        # Encoder
        self.block1 = Block1(1, 64)
        self.block2 = Block1(64, 64, stride = 2)
        self.block3 = Block1(64,128, stride = 2)

        #Bottom from U
        self.bottom_block = nn.Sequential(Block1(128, 256), Block1(256, 256, stride = 2))

        #Decoder
        self.dec_block_1 = nn.Sequential(Block2(384, 384), Block1(384, 128))
        self.dec_block_2 = nn.Sequential(Block2(192, 192), Block1(192, 64))

        self.dec_block_3 = nn.Sequential(Block2(128, 128), Block1(128, 32))
        self.final = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.bottom_block(x3)
        x4_interpol = F.interpolate(x4, size=x3.shape[2:], mode='bilinear', align_corners = False)

        cat_1 = torch.cat([x4_interpol, x3], dim=1)
        x5 = self.dec_block_1(cat_1)

        x5_interpol = F.interpolate(x5, size=x2.shape[2:], mode='bilinear', align_corners = False)

        cat_2 = torch.cat([x5_interpol, x2], dim=1)
        x6 = self.dec_block_2(cat_2)

        x6_interpol = F.interpolate(x6, size=x1.shape[2:], mode='bilinear', align_corners = False)

        cat_3 = torch.cat([x6_interpol, x1], dim=1)
        x7 = self.dec_block_3(cat_3)
        x8 = self.final(x7)
        return x8
