import torch
import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self,input_channels):
        super(Discriminator,self).__init__()
        kernel_size=4
        padding=1
        
        self.layer1=nn.Conv2d(input_channels,64,kernel_size=kernel_size,stride=2,padding=padding)
        
        self.layer2=nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True), 
            nn.Conv2d(64,128,kernel_size=kernel_size,stride=2,padding=padding),
            nn.BatchNorm2d(128))
        
        self.layer3=nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True), 
            nn.Conv2d(128,256,kernel_size=kernel_size,stride=2,padding=padding),
            nn.BatchNorm2d(256))
        
        self.layer4=nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True), 
            nn.Conv2d(256,512,kernel_size=kernel_size,stride=1,padding=padding),
            nn.BatchNorm2d(512))

        self.layer5=nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(512,1,kernel_size=kernel_size,stride=1,padding=padding))

        
    def forward(self,x,label):
        x = torch.cat([x, label],1)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.layer5(x)
        return nn.Sigmoid()(x)