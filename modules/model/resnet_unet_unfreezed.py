import torch
import torch.nn as nn
import torchvision.models as models


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        concat_channels = (in_channels // 2) + skip_channels
        
        self.conv = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ResNetUNetUnfreezed(nn.Module):
    def __init__(self, n_classes=1, unfreeze_all=True):
        super(ResNetUNetUnfreezed, self).__init__()
        
        # --- ENCODER: ResNet50 (Now Trainable) ---
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        self.first_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) 
        self.pool = resnet.maxpool
        self.enc1 = resnet.layer1 
        self.enc2 = resnet.layer2 
        self.enc3 = resnet.layer3 
        self.enc4 = resnet.layer4 

        # --- DECODER ---
        self.dec1 = DecoderBlock(2048, 1024, 512)
        self.dec2 = DecoderBlock(512, 512, 256)
        self.dec3 = DecoderBlock(256, 256, 128)
        self.dec4 = DecoderBlock(128, 64, 64)

        # --- FINAL HEAD ---
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.first_conv(x) # [B, 64, 224, 224]
        x_p = self.pool(x0) # [B, 64, 112, 112]
        x1 = self.enc1(x_p) # [B, 256, 112, 112]
        x2 = self.enc2(x1) # [B, 512, 56, 56]
        x3 = self.enc3(x2) # [B, 1024, 28, 28]
        x4 = self.enc4(x3) # [B, 2048, 14, 14]

        # Decoder
        d1 = self.dec1(x4, x3) # [B, 512, 28, 28]
        d2 = self.dec2(d1, x2) # [B, 256, 56, 56]
        d3 = self.dec3(d2, x1) # [B, 128, 112, 112]
        d4 = self.dec4(d3, x0) # [B, 64, 224, 224]
        
        # Final Head
        out = torch.relu(self.final_conv(self.final_up(d4))) 
        mask_logits = self.outc(out) 

        # Weakly Supervised Max-Pooling
        class_logits = torch.max(mask_logits.view(mask_logits.size(0), -1), dim=1, keepdim=True)[0]

        return mask_logits, class_logits
