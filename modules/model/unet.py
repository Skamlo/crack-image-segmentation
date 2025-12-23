import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Total ≈ 31,040,000 (roughly 31M parameters)
    """
    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.conv(x)

    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        
        # --- ENCODER ---
        # Parameters: ((3*3*3+1)*64) + ((3*3*64+1)*64) ≈ 38,720
        self.inc = DoubleConv(n_channels, 64) 
        
        # Parameters: ((3*3*64+1)*128) + ((3*3*128+1)*128) ≈ 221,440
        self.down1 = DoubleConv(64, 128) 
        
        # Parameters: ((3*3*128+1)*256) + ((3*3*256+1)*256) ≈ 885,248
        self.down2 = DoubleConv(128, 256)
        
        # Parameters: ((3*3*256+1)*512) + ((3*3*512+1)*512) ≈ 3,539,968
        self.down3 = DoubleConv(256, 512)
        
        # Parameters: ((3*3*512+1)*1024) + ((3*3*1024+1)*1024) ≈ 14,157,824
        self.down4 = DoubleConv(512, 1024)
        
        self.pool = nn.MaxPool2d(2)

        # --- DECODER ---
        # Parameters: (2*2*1024+1)*512 ≈ 2,097,664
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        # Parameters: (Concat 512+512): ≈ 7,079,936
        self.conv_up1 = DoubleConv(1024, 512)
        
        # Parameters: (2*2*512+1)*256 ≈ 524,544
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # Parameters: (Concat 256+256): ≈ 1,770,496
        self.conv_up2 = DoubleConv(512, 256)
        
        # Parameters: (2*2*256+1)*128 ≈ 131,200
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # Parameters: (Concat 128+128): ≈ 442,880
        self.conv_up3 = DoubleConv(256, 128)
        
        # Parameters: (2*2*128+1)*64 ≈ 32,832
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # Parameters: (Concat 64+64): ≈ 110,720
        self.conv_up4 = DoubleConv(128, 64)

        # Parameters: (1*1*64+1)*1 = 65
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Input: [B, 3, 448, 448]
        
        # Encoder
        x1 = self.inc(x) # Shape: [B, 64, 448, 448]
        x2 = self.down1(self.pool(x1)) # Shape: [B, 128, 224, 224]
        x3 = self.down2(self.pool(x2)) # Shape: [B, 256, 112, 112]
        x4 = self.down3(self.pool(x3)) # Shape: [B, 512, 56, 56]
        x5 = self.down4(self.pool(x4)) # Shape: [B, 1024, 28, 28]

        # Decoder
        u1 = self.up1(x5) # Shape: [B, 512, 56, 56]
        u1 = self.conv_up1(torch.cat([x4, u1], dim=1)) # Shape: [B, 512, 56, 56]
        
        u2 = self.up2(u1) # Shape: [B, 256, 112, 112]
        u2 = self.conv_up2(torch.cat([x3, u2], dim=1)) # Shape: [B, 256, 112, 112]
        
        u3 = self.up3(u2) # Shape: [B, 128, 224, 224]
        u3 = self.conv_up3(torch.cat([x2, u3], dim=1)) # Shape: [B, 128, 224, 224]
        
        u4 = self.up4(u3) # Shape: [B, 64, 448, 448]
        u4 = self.conv_up4(torch.cat([x1, u4], dim=1)) # Shape: [B, 64, 448, 448]

        # Output Segmentation Map
        mask_logits = self.outc(u4) # Shape: [B, 1, 448, 448]

        # Weak Supervision Head (Global Max Pooling)
        # Flatten [B, 1, 448, 448] -> [B, 200704]
        flat = mask_logits.view(mask_logits.size(0), -1)
        class_logits, _ = torch.max(flat, dim=1, keepdim=True) # [B, 1]

        return mask_logits, class_logits
