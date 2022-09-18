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

class GreenBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GreenBlock,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)

class segRAVIR(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 3, features = [16,32,64,128,256,256,256]):
        super(segRAVIR, self).__init__()
        self.downs = nn.ModuleList()
        self.ups_1 = nn.ModuleList()
        self.ups_2 = nn.ModuleList()
        self.green = nn.ModuleList()
        self.sky_blue = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #green
        self.green.append(GreenBlock(1,16))
        self.green.append(GreenBlock(16,16))
        self.green.append(GreenBlock(16,16))

        #sky-blue
        for i in range(0,4):
            self.sky_blue.append(nn.Conv2d(features[i],features[i+1],kernel_size=2,stride=2))
        
        
        for feature in features:
            self.downs.append(ResBlock(feature, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    def forward(self,x):
        skip_connections = []

        x = self.green[0](x)
        

        for i in range(0,4):
            x = self.downs[i](x)
            skip_connections.append(x)
            x = self.sky_blue[i](x)
        x = self.downs[4](x)
        x = self.downs[5](x)
        x = self.downs[6](x)
        skip_connections.append(x)
        



        


        




