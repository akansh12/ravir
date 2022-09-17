from turtle import forward
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(ResBlock,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(inplace=True), ###check_again for dropout and bias

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(inplace=True) ### Check_again
        )
    def forward(self,x):
        return x+self.conv(x)





