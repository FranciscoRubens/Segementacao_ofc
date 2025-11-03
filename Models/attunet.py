import torch
import torch.nn as nn
import torch.nn.functional as F

# BLOCO CONVOLUCIONAL
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# ATTENTION GATE
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape != x1.shape:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# ATTENTION U-NET
class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dconv_down1 = DoubleConv(1, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.att3 = AttentionGate(512, 256, 128)
        self.dconv_up3 = DoubleConv(512 + 256, 256)

        self.att2 = AttentionGate(256, 128, 64)
        self.dconv_up2 = DoubleConv(256 + 128, 128)

        self.att1 = AttentionGate(128, 64, 32)
        self.dconv_up1 = DoubleConv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)

        x = self.upsample(conv4)
        att3 = self.att3(x, conv3)
        x = torch.cat([x, att3], dim=1)
        x = self.dconv_up3(x)

        x = self.upsample(x)
        att2 = self.att2(x, conv2)
        x = torch.cat([x, att2], dim=1)
        x = self.dconv_up2(x)

        x = self.upsample(x)
        att1 = self.att1(x, conv1)
        x = torch.cat([x, att1], dim=1)
        x = self.dconv_up1(x)

        return torch.sigmoid(self.conv_last(x))