import torch
import torch.nn as nn
import torch.nn.functional as F 

# === Blocos da U-Net++ ===
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
    def forward(self, x):
        return self.up(x)

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=64, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        f = base_filters

        # Encoder
        self.conv00 = ConvBlock(in_channels, f)
        self.conv10 = ConvBlock(f, f*2)
        self.conv20 = ConvBlock(f*2, f*4)
        self.conv30 = ConvBlock(f*4, f*8)
        self.conv40 = ConvBlock(f*8, f*16)
        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.conv01 = ConvBlock(f+f, f)
        self.conv11 = ConvBlock(f*2 + f*2, f*2)
        self.conv21 = ConvBlock(f*4 + f*4, f*4)
        self.conv31 = ConvBlock(f*8 + f*8, f*8)

        self.conv02 = ConvBlock(f+f+f, f)
        self.conv12 = ConvBlock(f*2 + f*2 + f*2, f*2)
        self.conv22 = ConvBlock(f*4 + f*4 + f*4, f*4)

        self.conv03 = ConvBlock(f+f+f+f, f)
        self.conv13 = ConvBlock(f*2 + f*2 + f*2 + f*2, f*2)

        self.conv04 = ConvBlock(f+f+f+f+f, f)

        # Upsample
        self.up10 = UpBlock(f*2, f)
        self.up20 = UpBlock(f*4, f*2)
        self.up30 = UpBlock(f*8, f*4)
        self.up40 = UpBlock(f*16, f*8)

        # Saídas para supervisão profunda
        self.final1 = nn.Conv2d(f, out_channels, kernel_size=1)
        self.final2 = nn.Conv2d(f, out_channels, kernel_size=1)
        self.final3 = nn.Conv2d(f, out_channels, kernel_size=1)
        self.final4 = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool(x00))
        x20 = self.conv20(self.pool(x10))
        x30 = self.conv30(self.pool(x20))
        x40 = self.conv40(self.pool(x30))

        # Decoder
        x01 = self.conv01(torch.cat([x00, self.up10(x10)], dim=1))
        x11 = self.conv11(torch.cat([x10, self.up20(x20)], dim=1))
        x21 = self.conv21(torch.cat([x20, self.up30(x30)], dim=1))
        x31 = self.conv31(torch.cat([x30, self.up40(x40)], dim=1))

        x02 = self.conv02(torch.cat([x00, x01, self.up10(x11)], dim=1))
        x12 = self.conv12(torch.cat([x10, x11, self.up20(x21)], dim=1))
        x22 = self.conv22(torch.cat([x20, x21, self.up30(x31)], dim=1))

        x03 = self.conv03(torch.cat([x00, x01, x02, self.up10(x12)], dim=1))
        x13 = self.conv13(torch.cat([x10, x11, x12, self.up20(x22)], dim=1))

        x04 = self.conv04(torch.cat([x00, x01, x02, x03, self.up10(x13)], dim=1))

        if self.deep_supervision:
            return [
                torch.sigmoid(self.final1(x01)),
                torch.sigmoid(self.final2(x02)),
                torch.sigmoid(self.final3(x03)),
                torch.sigmoid(self.final4(x04)),
            ]
        else:
            return torch.sigmoid(self.final4(x04))