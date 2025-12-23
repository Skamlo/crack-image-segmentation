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


class ResNetUNet(nn.Module):
    """
    Total Frozen Params (ResNet 50): 23,508,032
    Total Trainable Params: 20,859,297
    """
    def __init__(self, n_classes=1):
        super(ResNetUNet, self).__init__()
        
        # --- ENCODER: ResNet50 (Frozen) ---
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.first_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # 9,536 params
        self.pool = resnet.maxpool
        self.enc1 = resnet.layer1 # 215,808 params
        self.enc2 = resnet.layer2 # 1,219,584 params
        self.enc3 = resnet.layer3 # 7,098,368 params
        self.enc4 = resnet.layer4 # 14,964,736 params

        for param in self.parameters():
            param.requires_grad = False

        # --- DECODER (Trainable) ---
        # Dec1: Up(2048->1024) + Conv(2048->512)
        # Params: 8,389,632 (Up) + 9,438,720 (Conv+BNs) = 17,828,352
        self.dec1 = DecoderBlock(2048, 1024, 512)
        
        # Dec2: Up(512->256) + Conv(768->256)
        # Params: 524,544 (Up) + 1,770,752 (Conv+BNs) = 2,295,296
        self.dec2 = DecoderBlock(512, 512, 256)
        
        # Dec3: Up(256->128) + Conv(384->128)
        # Params: 131,200 (Up) + 443,136 (Conv+BNs) = 574,336
        self.dec3 = DecoderBlock(256, 256, 128)
        
        # Dec4: Up(128->64) + Conv(128->64)
        # Params: 32,832 (Up) + 110,976 (Conv+BNs) = 143,808
        self.dec4 = DecoderBlock(128, 64, 64)

        # --- FINAL HEAD (Trainable) ---
        # final_up: (2*2 * 64 * 32) + 32 = 8,224
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        # final_conv: (3*3 * 32 * 32) + 32 = 9,248
        self.final_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # outc: (1*1 * 32 * 1) + 1 = 33
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.first_conv(x) # Shape: [B, 64, 224, 224]
        x_p = self.pool(x0) # Shape: [B, 64, 112, 112]
        x1 = self.enc1(x_p) # Shape: [B, 256, 112, 112]
        x2 = self.enc2(x1) # Shape: [B, 512, 56, 56]
        x3 = self.enc3(x2) # Shape: [B, 1024, 28, 28]
        x4 = self.enc4(x3) # Shape: [B, 2048, 14, 14]

        # Decoder
        d1 = self.dec1(x4, x3) # Shape: [B, 512, 28, 28]
        d2 = self.dec2(d1, x2) # Shape: [B, 256, 56, 56]
        d3 = self.dec3(d2, x1) # Shape: [B, 128, 112, 112]
        d4 = self.dec4(d3, x0) # Shape: [B, 64, 224, 224]
        
        # Final Head
        out = torch.relu(self.final_conv(self.final_up(d4))) # Shape: [B, 32, 448, 448]
        mask_logits = self.outc(out) # Shape: [B, 1, 448, 448]

        # Weakly Supervised Max-Pooling
        class_logits = torch.max(mask_logits.view(mask_logits.size(0), -1), dim=1, keepdim=True)[0]

        return mask_logits, class_logits
