import torch
import torch.nn as nn

class encoder_block(nn.Module):
    def __init__(self, in_ch, out_ch, act=True, batch_norm=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size = 4, stride = 2, padding = 1)
        self.relu = nn.LeakyReLU(0.2, True)
        self.bn = batch_norm
        self.act = act
        self.batchnorm = nn.BatchNorm2d(out_ch)
        
        
    def forward(self, x):
        if self.act:
            x = self.conv(self.relu(x))
        else:
            x = self.conv(x)

        if self.bn:
            return self.batchnorm(x)
        else:
            return x  

class decoder_block(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False, batch_norm=True):
        #super(decoder_block, __self__).__init__()
        super().__init__()

        self.transposeconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.bn = batch_norm
        self.batchnorm = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(True)
        self.dropout = dropout
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        if self.bn:
            x = self.transposeconv(self.relu(x))
            x = self.batchnorm(x)
        else:
            x = self.transposeconv(self.relu(x))

        if self.dropout:
            return self.drop(x)
        else:
            return x


class Generator(nn.Module):
    def __init__(self, in_ch, filters, out_ch,**kwargs):
        #super(Generator, self).__init__()
        super().__init__()

        self.conv1 = encoder_block(in_ch, filters, act=False, batch_norm=False)
        self.conv2 = encoder_block(filters, filters*2)
        self.conv3 = encoder_block(filters*2, filters*4)
        self.conv4 = encoder_block(filters*4, filters*8)
        self.conv5 = encoder_block(filters*8, filters*8)
        self.conv6 = encoder_block(filters*8, filters*8)
        self.conv7 = encoder_block(filters*8, filters*8)
        self.conv8 = encoder_block(filters*8, filters*8, batch_norm=False)
        
        self.trconv1 = decoder_block(filters*8, filters*8, dropout=False)
        self.trconv2 = decoder_block(filters*8*2, filters*8, dropout=True)
        self.trconv3 = decoder_block(filters*8*2, filters*8, dropout=True)
        self.trconv4 = decoder_block(filters*8*2, filters*8)
        self.trconv5 = decoder_block(filters*8*2, filters*4)
        self.trconv6 = decoder_block(filters*4*2, filters*2)
        self.trconv7 = decoder_block(filters*2*2, filters)
        self.trconv8 = decoder_block(filters*2, out_ch, batch_norm=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)

        dec1 = torch.cat([self.trconv1(x8), x7],1)
        dec2 = torch.cat([self.trconv2(dec1), x6],1)
        dec3 = torch.cat([self.trconv3(dec2), x5],1)
        dec4 = torch.cat([self.trconv4(dec3), x4],1)
        dec5 = torch.cat([self.trconv5(dec4), x3],1)
        dec6 = torch.cat([self.trconv6(dec5), x2],1)
        dec7 = torch.cat([self.trconv7(dec6), x1],1)
        dec8 = self.trconv8(dec7)

        return nn.Tanh()(dec8)