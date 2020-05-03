import torch.nn as nn
from nnBlocks import *
from torchvision import models

        
class FullConvBase(nn.Module):
    
    def __init__(self, to_clamp=True):
        
        super().__init__()
        
        self.block = nn.Sequential(
            ConvolutionalBlock(in_channels=3,  out_channels=64, activation='lrelu'), 
            ConvolutionalBlock(in_channels=64, out_channels=64, activation='lrelu'))
        self.upsample = UpsampleBlock(in_channels=64, to_clamp=to_clamp)

    
    def forward(self, x, return_upsample=False):
        
        out = self.block(x)
        img, out  = self.upsample(out)
    
        if return_upsample:
            return img, out
        return img
    

class FullConvLarge(nn.Module):
    def __init__(self):
        
        super(FullConvLarge, self).__init__()
        
        self.block1 = FullConvBase()
        self.upsample = UpsampleBlock(in_channels=64)

    
    def forward(self, x):
        
        imgM, out = self.block1(x, return_upsample=True)
        imgL, _ = self.upsample(out)
        
        return imgM, imgL
    

class ResidualNet(nn.Module):
    
    def __init__(self):
        
        super(ResidualNet, self).__init__()
        
        self.block = nn.Sequential(
            ConvolutionalBlock(in_channels=3, out_channels=32, kernel_size=1, padding=0, activation='lrelu'), 
            ResidualBlock(), 
            ResidualBlock()
        )
        self.upsample1 = UpsampleBlock(in_channels=32)
        self.residual3 = ResidualBlock()
        self.upsample2 = UpsampleBlock(in_channels=32)
        
    
    def forward(self, x):
        out = self.block(x)
        imgM, out = self.upsample1(out)
        out = self.residual3(out)
        imgL, _ = self.upsample2(out)
        
        return imgM, imgL
    

class DilatedNet(nn.Module):
    
    def __init__(self):
        
        super(DilatedNet, self).__init__()
        
        self.block = nn.Sequential(
            ConvolutionalBlock(in_channels=3, out_channels=32, kernel_size=1, padding=0, activation='lrelu'), 
            DilatedBlock(), 
            DilatedBlock()
        )
        self.upsample1 = UpsampleBlock(in_channels=32)
        self.dilated3 = DilatedBlock()
        self.upsample2 = UpsampleBlock(in_channels=32)
        
    
    def forward(self, x):
        out = self.block(x)
        imgM, out = self.upsample1(out)
        out = self.dilated3(out)
        imgL, _ = self.upsample2(out)
        
        return imgM, imgL


class PreTrainedNet(nn.Module):
    
    def __init__(self, to_shuffle=False):
        
        super(PreTrainedNet, self).__init__()
        
        self.to_shuffle = to_shuffle
        self.feature_extractor = VGG16()
        
        self.block = nn.Sequential(
            ConvolutionalBlock(in_channels=3,  out_channels=64, activation='lrelu'), 
            ConvolutionalBlock(in_channels=64, out_channels=64, activation='lrelu'))
        
        fan_in = 128
        self.upsample1 = UpsampleBlock(in_channels=fan_in, to_shuffle=self.to_shuffle)
        
        if self.to_shuffle:
            fan_in //= 4
        self.upsample2 = UpsampleBlock(in_channels=fan_in, to_shuffle=self.to_shuffle)

    
    def forward(self, x):
        
        features = self.feature_extractor(x)
        out = self.block(x)
        out = torch.cat([out, features], 1)
        imgM, out  = self.upsample1(out)
        imgL, _  = self.upsample2(out)
        
        return imgM, imgL
    
    
    
    
    
    
    