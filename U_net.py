import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Blocul de bază al U-Net: două convoluții consecutive cu ReLU.
    Acest bloc este folosit în toate nivelurile rețelei pentru
    extragerea de caracteristici.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            # Prima convoluție 3x3 cu padding pentru păstrarea dimensiunii
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),  # Activare ReLU pentru non-linearitate
            # A doua convoluție 3x3 
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """
    Blocul de downsampling din encoder-ul U-Net.
    Folosește MaxPooling pentru reducerea dimensiunii spatiale
    și DoubleConv pentru extragerea caracteristicilor.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            # MaxPool2d reduce dimensiunea cu factorul 2
            nn.MaxPool2d(2),
            # Extrage caracteristici la noua rezoluție
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    """
    Blocul de upsampling din decoder-ul U-Net.
    Mărește dimensiunea spațială și combină caracteristicile
    de la nivelul curent cu cele de la skip connection.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # ConvTranspose2d pentru upsampling (mărirea dimensiunii)
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        # DoubleConv pentru procesarea caracteristicilor concatenate
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        x1: caracteristici de la nivelul inferior (upsampled)
        x2: caracteristici de la skip connection (encoder)
        """
        # Upsampling pentru x1
        x1 = self.up(x1)
        
        # Calculează diferențele de dimensiune pentru padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        # Padding pentru a face dimensiunile compatibile
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        # Concatenează caracteristicile pe dimensiunea canalelor
        x = torch.cat([x2, x1], dim=1)
        
        # Procesează caracteristicile concatenate
        return self.conv(x)

class OutConv(nn.Module):
    """
    Stratul final de output al U-Net.
    Convertește caracteristicile finale în predicții pentru fiecare clasă.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Convoluție 1x1 pentru clasificarea finală
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    Implementarea arhitecturii U-Net pentru segmentarea imaginilor.
    
    U-Net constă din:
    1. Encoder (contracting path): extrage caracteristici prin downsampling
    2. Bottleneck: nivelul cel mai profund
    3. Decoder (expanding path): reconstruiește imaginea prin upsampling
    4. Skip connections: conectează encoder-ul cu decoder-ul
    
    Args:
        n_channels (int): numărul de canale de input (ex: 3 pentru RGB)
        n_classes (int): numărul de clase pentru segmentare
    """
    def __init__(self, n_channels, n_classes):
        super().__init__()
        # Input convolution (primul nivel)
        self.inc = DoubleConv(n_channels, 64)
        
        # Encoder path (downsampling)
        self.down1 = Down(64, 128)    # 64 → 128 canale
        self.down2 = Down(128, 256)   # 128 → 256 canale  
        self.down3 = Down(256, 512)   # 256 → 512 canale
        self.down4 = Down(512, 1024)  # 512 → 1024 canale (bottleneck)
        
        # Decoder path (upsampling)
        self.up1 = Up(1024, 512)      # 1024 → 512 canale
        self.up2 = Up(512, 256)       # 512 → 256 canale
        self.up3 = Up(256, 128)       # 256 → 128 canale
        self.up4 = Up(128, 64)        # 128 → 64 canale
        
        # Output convolution (clasificarea finală)
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x):
        """
        Forward pass prin rețeaua U-Net.
        
        Args:
            x (Tensor): imaginea de input cu shape (batch_size, n_channels, H, W)
            
        Returns:
            Tensor: logits pentru fiecare clasă cu shape (batch_size, n_classes, H, W)
        """
        # Encoder path cu salvarea caracteristicilor pentru skip connections
        x1 = self.inc(x)      # Input level
        x2 = self.down1(x1)   # 1/2 resolution
        x3 = self.down2(x2)   # 1/4 resolution  
        x4 = self.down3(x3)   # 1/8 resolution
        x5 = self.down4(x4)   # 1/16 resolution (bottleneck)
        
        # Decoder path cu skip connections
        x = self.up1(x5, x4)  # Combină bottleneck cu x4
        x = self.up2(x, x3)   # Combină cu x3
        x = self.up3(x, x2)   # Combină cu x2
        x = self.up4(x, x1)   # Combină cu x1
        
        # Output final
        return self.outc(x)
