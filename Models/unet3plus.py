import torch
import torch.nn as nn
import torch.nn.functional as F

# ARQUITETURA UNET 3+
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet3Plus(nn.Module):
    def __init__(self, in_channels=1, n_classes=1, base_ch=64, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        chs = [base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16]

        # Encoder
        self.encoder1 = ConvBlock(in_channels, chs[0])
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = ConvBlock(chs[0], chs[1])
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = ConvBlock(chs[1], chs[2])
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = ConvBlock(chs[2], chs[3])
        self.pool4 = nn.MaxPool2d(2)
        self.encoder5 = ConvBlock(chs[3], chs[4])

        # 1x1 convoluções
        self.conv1x1 = nn.ModuleList([
            nn.Conv2d(chs[i], base_ch, kernel_size=1) for i in range(5)
        ])

        # Decodificadores
        def make_decoder_block():
            return nn.Sequential(
                nn.Conv2d(base_ch * 5, base_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_ch),
                nn.ReLU(inplace=True)
            )

        self.decoder4 = make_decoder_block()
        self.decoder3 = make_decoder_block()
        self.decoder2 = make_decoder_block()
        self.decoder1 = make_decoder_block()

        self.final = nn.ModuleList([
            nn.Conv2d(base_ch, n_classes, kernel_size=1) for _ in range(4)
        ])

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))
        e5 = self.encoder5(self.pool4(e4))

        # 1x1 convoluções
        e1_ = self.conv1x1[0](e1)
        e2_ = self.conv1x1[1](e2)
        e3_ = self.conv1x1[2](e3)
        e4_ = self.conv1x1[3](e4)
        e5_ = self.conv1x1[4](e5)

        def upsample_to(src, tgt):
            return F.interpolate(src, size=tgt.shape[2:], mode='bilinear', align_corners=True)

        # Decoder
        d4 = self.decoder4(torch.cat([
            upsample_to(e1_, e4),
            upsample_to(e2_, e4),
            upsample_to(e3_, e4),
            e4_,
            upsample_to(e5_, e4)
        ], dim=1))

        d3 = self.decoder3(torch.cat([
            upsample_to(e1_, e3),
            upsample_to(e2_, e3),
            e3_,
            upsample_to(e4_, e3),
            upsample_to(e5_, e3)
        ], dim=1))

        d2 = self.decoder2(torch.cat([
            upsample_to(e1_, e2),
            e2_,
            upsample_to(e3_, e2),
            upsample_to(e4_, e2),
            upsample_to(e5_, e2)
        ], dim=1))

        d1 = self.decoder1(torch.cat([
            e1_,
            upsample_to(e2_, e1),
            upsample_to(e3_, e1),
            upsample_to(e4_, e1),
            upsample_to(e5_, e1)
        ], dim=1))

        if self.deep_supervision:
            out1 = self.final[0](d1)
            out2 = upsample_to(self.final[1](d2), x)
            out3 = upsample_to(self.final[2](d3), x)
            out4 = upsample_to(self.final[3](d4), x)
            return [torch.sigmoid(out1), torch.sigmoid(out2), torch.sigmoid(out3), torch.sigmoid(out4)]
        else:
            return torch.sigmoid(self.final[0](d1))