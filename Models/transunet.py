import torch
import torch.nn as nn
import torchvision.models as models



# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ViT Bottleneck

class ViT(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim)
            for _ in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# Decoder Block

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# TransUNet 
class TransUNet_ResNet50(nn.Module):
    def __init__(self,
                 in_channels=1,
                 num_classes=1,
                 base_channels=32,
                 vit_depth=4,
                 vit_heads=4,
                 vit_mlp_dim=256):
        super().__init__()

        # ResNet-50 Encoder 
        resnet = models.resnet50(weights=None)

        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1   
        self.layer2 = resnet.layer2   
        self.layer3 = resnet.layer3   
        self.layer4 = resnet.layer4   

        # Channel Reduction 
        self.reduce4 = nn.Conv2d(2048, base_channels * 8, 1)
        self.reduce3 = nn.Conv2d(1024, base_channels * 4, 1)
        self.reduce2 = nn.Conv2d(512,  base_channels * 2, 1)
        self.reduce1 = nn.Conv2d(256,  base_channels, 1)

        #  ViT 
        self.vit = ViT(
            dim=base_channels * 8,
            depth=vit_depth,
            heads=vit_heads,
            mlp_dim=vit_mlp_dim
        )

        # Decoder 
        self.dec3 = DecoderBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.dec2 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.dec1 = DecoderBlock(base_channels * 2, base_channels, base_channels)

        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.out_conv = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)              
        x = self.maxpool(x1)

        x2 = self.layer1(x)            
        x3 = self.layer2(x2)           
        x4 = self.layer3(x3)           
        x5 = self.layer4(x4)           

        # Channel reduction
        x1 = self.reduce1(x2)
        x2 = self.reduce2(x3)
        x3 = self.reduce3(x4)
        x4 = self.reduce4(x5)

        # ViT
        b, c, h, w = x4.shape
        x4_flat = x4.flatten(2).transpose(1, 2)
        x4_flat = self.vit(x4_flat)
        x4 = x4_flat.transpose(1, 2).reshape(b, c, h, w)

        # Decoder
        d3 = self.dec3(x4, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)

        out = self.final_up(d1)
        return self.out_conv(out)

