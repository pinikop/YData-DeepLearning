import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from torchvision.transforms import Normalize


class ConvolutionalBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, padding=1, stride=1, dilation=1,
                 activation=None, batch_norm=False):
        
        super(ConvolutionalBlock, self).__init__()
        
        layers = list()
        
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, padding=padding, 
                                stride=stride, dilation=dilation))
        
        if batch_norm:
            layers.append(nn.BatchNorm2d(num_features=out_channels))
           
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'lrelu':
            layers.append(nn.LeakyReLU())
         
        self.conv_block = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.conv_block(x)
        return out
    
    

class UpsampleBlock(nn.Module):
    
    def __init__(self, in_channels, scale_factor=2, to_shuffle=False, 
                 mode='bilinear', align_corners=False, to_clamp=False):
        
        super(UpsampleBlock, self).__init__()
        self.to_shuffle = to_shuffle
        self.scale_factor = scale_factor
        self.to_clamp = to_clamp
        
        if self.to_shuffle:
            self.upsample = nn.PixelShuffle(upscale_factor=self.scale_factor)
            self.in_channels = in_channels // self.scale_factor ** 2
        
        else:
            self.mode = mode
            self.align_corners = align_corners
            self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode=self.mode,
                                        align_corners=self.align_corners)
            self.in_channels = in_channels 
        
        self.conv_block = nn.Sequential(
            ConvolutionalBlock(in_channels=self.in_channels, out_channels=3,
                               kernel_size=1, padding=0), 
            nn.Sigmoid()
        )        
    
    
    def forward(self, x):
        
        out = self.upsample(x)
        img = self.conv_block(out)

        if self.to_clamp:
            img = torch.clamp(img, min=0., max=1.)

        return img, out
    
    
class ResidualBase(nn.Module):
    
    def __init__(self, block):
        
        super(ResidualBase, self).__init__()
        self.block = block
    
    def forward(self, x):
        return self.block(x) + x
    

class ResidualBlock(nn.Module):
    
    def __init__(self):
        
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ResidualBase(
                nn.Sequential(
                    ConvolutionalBlock(32, 32, activation='lrelu'),
                    ConvolutionalBlock(32, 32, activation='lrelu')
                )
            ),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.block(x)
    

class DilatedBlock(nn.Module):
    
    def __init__(self, dilations=[1, 2, 4]):
        
        super(DilatedBlock, self).__init__()
        self.dilated = nn.ModuleList([ConvolutionalBlock(32, 32, dilation=d, padding=d, activation='lrelu') for d in dilations])
        self.conv_block = ConvolutionalBlock(96, 32, activation='lrelu')
    
    def forward(self, x):
        
        out = torch.cat([d(x) for d in self.dilated], dim=1)
        out = F.relu(out)
        out = self.conv_block(out)
    
        return out

    
class VGG16(nn.Module):
    
    def __init__(self):
        
        super(VGG16, self).__init__()
        
        self.model = vgg16(pretrained=True).features[:4]
        for p in self.model.parameters():
            p.requires_grad = False
        
        self.transform = Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]
                                  )
    
    def forward(self, x):
        x = torch.cat([self.transform(x[i]).unsqueeze(0) for i in range(x.size(0))])
        return self.model(x)


