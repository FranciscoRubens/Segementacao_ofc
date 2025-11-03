import torch
import torch.nn as nn
import torch.nn.functional as F

# BLOCO CONVOLUCIONAL
class Conv3x3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class SeparableConv3x3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

# ===============================
# BLOCO UNET COM 2 CONVOLUÇÕES
# ===============================
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, separable=False):
        super().__init__()
        conv = SeparableConv3x3 if separable else Conv3x3
        self.double_conv = nn.Sequential(
            conv(in_ch, out_ch),
            conv(out_ch, out_ch)
        )
    def forward(self, x):
        return self.double_conv(x)

# ===============================
# W-NET
# ===============================
class WNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, separable=False):
        super().__init__()

        # UNET 1 (segmentação)
        self.enc1_1 = UNetBlock(in_ch, 64, separable)
        self.enc2_1 = UNetBlock(64, 128, separable)
        self.enc3_1 = UNetBlock(128, 256, separable)
        self.enc4_1 = UNetBlock(256, 512, separable)
        self.bottleneck_1 = UNetBlock(512, 1024, separable)

        self.up4_1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4_1 = UNetBlock(1024, 512, separable)
        self.up3_1 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3_1 = UNetBlock(512, 256, separable)
        self.up2_1 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2_1 = UNetBlock(256, 128, separable)
        self.up1_1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1_1 = UNetBlock(128, 64, separable)

        self.final_seg = nn.Conv2d(64, out_ch, 1)

        # UNET 2 (reconstrução)
        self.enc1_2 = UNetBlock(out_ch, 64, separable)
        self.enc2_2 = UNetBlock(64, 128, separable)
        self.enc3_2 = UNetBlock(128, 256, separable)
        self.enc4_2 = UNetBlock(256, 512, separable)
        self.bottleneck_2 = UNetBlock(512, 1024, separable)

        self.up4_2 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4_2 = UNetBlock(1024, 512, separable)
        self.up3_2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3_2 = UNetBlock(512, 256, separable)
        self.up2_2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2_2 = UNetBlock(256, 128, separable)
        self.up1_2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1_2 = UNetBlock(128, 64, separable)

        self.final_recon = nn.Conv2d(64, in_ch, 1)

    def forward(self, x):
        # UNET 1 - Segmentação
        c1 = self.enc1_1(x)
        p1 = F.max_pool2d(c1, 2)
        c2 = self.enc2_1(p1)
        p2 = F.max_pool2d(c2, 2)
        c3 = self.enc3_1(p2)
        p3 = F.max_pool2d(c3, 2)
        c4 = self.enc4_1(p3)
        p4 = F.max_pool2d(c4, 2)
        c5 = self.bottleneck_1(p4)
        u4 = self.up4_1(c5)
        u4 = torch.cat([u4, c4], 1)
        c6 = self.dec4_1(u4)
        u3 = self.up3_1(c6)
        u3 = torch.cat([u3, c3], 1)
        c7 = self.dec3_1(u3)
        u2 = self.up2_1(c7)
        u2 = torch.cat([u2, c2], 1)
        c8 = self.dec2_1(u2)
        u1 = self.up1_1(c8)
        u1 = torch.cat([u1, c1], 1)
        c9 = self.dec1_1(u1)
        seg = self.final_seg(c9)  # **sem sigmoid!**

        # UNET 2 - Reconstrução
        c1r = self.enc1_2(torch.sigmoid(seg))
        p1r = F.max_pool2d(c1r, 2)
        c2r = self.enc2_2(p1r)
        p2r = F.max_pool2d(c2r, 2)
        c3r = self.enc3_2(p2r)
        p3r = F.max_pool2d(c3r, 2)
        c4r = self.enc4_2(p3r)
        p4r = F.max_pool2d(c4r, 2)
        c5r = self.bottleneck_2(p4r)
        u4r = self.up4_2(c5r)
        u4r = torch.cat([u4r, c4r], 1)
        c6r = self.dec4_2(u4r)
        u3r = self.up3_2(c6r)
        u3r = torch.cat([u3r, c3r], 1)
        c7r = self.dec3_2(u3r)
        u2r = self.up2_2(c7r)
        u2r = torch.cat([u2r, c2r], 1)
        c8r = self.dec2_2(u2r)
        u1r = self.up1_2(c8r)
        u1r = torch.cat([u1r, c1r], 1)
        c9r = self.dec1_2(u1r)
        recon = torch.sigmoid(self.final_recon(c9r))  # reconstrução mantém sigmoid

        return seg, recon