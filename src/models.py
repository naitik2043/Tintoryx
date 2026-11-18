import torch
from torch import nn

def block(in_ch, out_ch, batch_norm=True):
    layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not batch_norm)]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

class UNetGenerator(nn.Module):
    """Compact U-Net: grayscale 1x32x32 -> RGB 3x32x32."""
    def __init__(self, base=32):
        super().__init__()
        self.down1 = block(1, base, False)       # 16x16
        self.down2 = block(base, base*2)         # 8x8
        self.down3 = block(base*2, base*4)       # 4x4
        self.down4 = block(base*4, base*8)       # 2x2
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base*8, base*8, 3, 1, 1),
            nn.ReLU(True)
        )
        self.up1 = nn.Sequential(nn.ConvTranspose2d(base*8, base*4, 4, 2, 1),
                                 nn.BatchNorm2d(base*4), nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(base*8, base*2, 4, 2, 1),
                                 nn.BatchNorm2d(base*2), nn.ReLU(True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(base*4, base, 4, 2, 1),
                                 nn.BatchNorm2d(base), nn.ReLU(True))
        self.up4 = nn.Sequential(nn.ConvTranspose2d(base*2, base, 4, 2, 1),
                                 nn.BatchNorm2d(base), nn.ReLU(True))
        self.out = nn.Sequential(nn.Conv2d(base, 3, 3, 1, 1), nn.Sigmoid())

    def forward(self, x):
        d1 = self.down1(x); d2 = self.down2(d1)
        d3 = self.down3(d2); d4 = self.down4(d3)
        z = self.bottleneck(d4)
        u1 = self.up1(z)
        u2 = self.up2(torch.cat([u1, d3], dim=1))
        u3 = self.up3(torch.cat([u2, d2], dim=1))
        u4 = self.up4(torch.cat([u3, d1], dim=1))
        return self.out(u4)

class PatchDiscriminator(nn.Module):
    """Conditional PatchGAN; input is concat(gray, candidate RGB) = 4 channels."""
    def __init__(self, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, base, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(base, base*2, 4, 2, 1), nn.BatchNorm2d(base*2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(base*2, base*4, 3, 1, 1), nn.BatchNorm2d(base*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(base*4, 1, 3, 1, 1)
        )

    def forward(self, gray, rgb):
        return self.net(torch.cat([gray, rgb], dim=1))
