import torch
import torch.nn as nn
import torch.nn.functional as F


# <----------------------------------------------- 3D-Unet Model ----------------------------------------------->

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UpsampleBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.upsample(x)

class UNet3D_Map(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, upsample_factor=2):
        super().__init__()
        self.upsample_factor = upsample_factor

        # Encoder
        self.enc1 = ConvBlock3D(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ConvBlock3D(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = ConvBlock3D(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock3D(base_channels * 4, base_channels * 8)

        # Decoder
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base_channels * 2, base_channels)

        self.upsample_final = UpsampleBlock3D(base_channels, base_channels, scale_factor=upsample_factor)
        self.output_layer = nn.Conv3d(base_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        # Decoder
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Final upsampling
        upsampled = self.upsample_final(d1)
        out = self.output_layer(upsampled)
        return self.sigmoid(out)
    



# <---------------------------------------------------- 3D-Uformer Model ---------------------------------------------------------->

class PatchEmbedding(torch.nn.Module):
    def __init__(self, embd_dim):
        super(PatchEmbedding).__init__()

        self.embd_dim = embd_dim
        
        def forward(self, x):
            shape = x.shape
            




# <------------------------------------------------------------- Test ------------------------------------------------------------->

if __name__ == "__main__":
    B, C, H, W, D = 2, 1, 128, 128, 64
    x = torch.randn(B, C, H, W, D)

    model = UNet3D_Map(in_channels=1)

    with torch.no_grad():
        reliability_map = model(x)

    threshold = 0.5
    corruption_map = (reliability_map > threshold).float()

    print(corruption_map.shape)




