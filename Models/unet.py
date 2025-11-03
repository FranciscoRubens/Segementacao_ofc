import torch
import torch.nn as nn
import torch.nn.functional as F

# Modelo UNet
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(DoubleConv, self).__init__()
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

class UNet(nn.Module):
    def __init__(self, num_classes=1):
        super(UNet, self).__init__()
        self.dconv_down1 = DoubleConv(1, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)
        self.dconv_bottom = DoubleConv(512, 1024)

        self.maxpool = nn.MaxPool2d(2)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dconv_up4 = DoubleConv(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dconv_up3 = DoubleConv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dconv_up2 = DoubleConv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dconv_up1 = DoubleConv(128, 64)

        self.conv_last = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        x = self.dconv_bottom(x)
        x = self.upconv4(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.dconv_up4(x)
        x = self.upconv3(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upconv2(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upconv1(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        if self.conv_last.out_channels == 1:
            out = torch.sigmoid(self.conv_last(x))
        else:
            out = torch.softmax(self.conv_last(x), dim=1)
        return out