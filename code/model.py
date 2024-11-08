import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, time_step):
        super(Attention, self).__init__()
        
        self.time_step = time_step
        self.fc1 = nn.Linear(time_step, 512)
        self.fc2 = nn.Linear(512, time_step)
        
    def forward(self, f):
        avg = torch.mean(f, dim=[1, 3, 4])
        max = f.max(dim=1).values.max(dim=2).values.max(dim=2).values
        
        a = torch.sigmoid(self.fc2(self.fc1(max)) + self.fc2(self.fc1(avg)))
        a = a.view(-1, 1, f.shape[2], 1, 1)
        f = torch.sum(a * f, dim=2)
        
        return f

class DisBlock2d(nn.Module):
    def __init__(self, channels):
        super(DisBlock2d, self).__init__()

        self.forward1 = nn.Sequential(
                nn.LeakyReLU(),
                nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),
                nn.AvgPool2d(2))

        self.forward2 = nn.Sequential(
                nn.Conv2d(channels[0], channels[2], kernel_size=1),
                nn.AvgPool2d(2))

    def forward(self, x):
        return self.forward1(x) + self.forward2(x)

class DisBlock3d(nn.Module):
    def __init__(self, channels):
        super(DisBlock3d, self).__init__()

        self.forward1 = nn.Sequential(
                nn.LeakyReLU(),
                nn.Conv3d(channels[0], channels[1], kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(channels[1], channels[2], kernel_size=3, padding=1))

        self.forward2 = nn.Sequential(
                nn.Conv3d(channels[0], channels[2], kernel_size=1))

    def forward(self, x):
        y = self.forward1(x)
        z = self.forward2(x)

        if x.size(2) > 1:
            y = F.avg_pool3d(y, 2)
            z = F.avg_pool3d(z, 2)
        else:
            y = F.avg_pool3d(y, (1, 2, 2))
            z = F.avg_pool3d(z, (1, 2, 2))

        return y + z

class DecBlock(nn.Module):
    def __init__(self, channels, scale_factor):
        super(DecBlock, self).__init__()
        
        self.forward1 = nn.Sequential(
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
                nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1))
        
        self.forward2 = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
                nn.Conv2d(channels[0], channels[2], kernel_size=1))
        
    def forward(self, x):
        return self.forward1(x) + self.forward2(x)

class EncBlock(nn.Module):
    def __init__(self, channels, scale_factor):
        super(EncBlock, self).__init__()

        self.scale_factor = scale_factor

        self.forward1 = nn.Sequential(
                nn.LeakyReLU(),
                nn.Conv3d(channels[0], channels[1], kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(channels[1], channels[2], kernel_size=3, padding=1))

        self.forward2 = nn.Sequential(
                nn.Conv3d(channels[0], channels[2], kernel_size=1))

    def forward(self, x):
        y = F.interpolate(self.forward1(x), scale_factor=self.scale_factor, mode='area')
        z = F.interpolate(self.forward2(x), scale_factor=self.scale_factor, mode='area')
        
        return y + z

class Discriminator(nn.Module):
    def __init__(self, block2d, block3d):
        super(Discriminator, self).__init__()

        self.forward1 = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1), # 600
                block2d([  8,   8,   16]), # 300
                block2d([ 16,  16,   32]), # 150
                block2d([ 32,  32,   64]), #  75
                block2d([ 64,  64,  128]), #  38
                block2d([128, 128,  256]), #  19
                block2d([256, 256,  512]), #  10
                block2d([512, 512, 1024]), #   5
                nn.LeakyReLU(inplace=True))

        self.forward2 = nn.Sequential(
                nn.Conv3d(1, 32, kernel_size=3, padding=1), # 120
                block3d([ 32,  32,   64]), # 60
                block3d([ 64,  64,  128]), # 30
                block3d([128, 128,  256]), # 15
                block3d([256, 256,  512]), #  8
                block3d([512, 512, 1024]), #  4
                nn.LeakyReLU(inplace=True))

        self.fc1 = nn.Linear(1024,    1)
        self.fc2 = nn.Linear(1024, 1024, bias=False)

    def forward(self, x, y):
        x = self.forward1(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.sum(x, dim=2)

        #######################

        y = self.forward2(y)
        y = y.view(y.size(0), y.size(1), -1)
        y = torch.sum(y, dim=2)

        return self.fc1(x) + torch.sum(x * self.fc2(y), dim=1, keepdim=True)

class Generator(nn.Module):
    def __init__(self, enc_block, dec_block):
        super(Generator, self).__init__()

        self.attention = nn.ModuleList()
        self.attention.append(Attention(8))
        self.attention.append(Attention(4))
        self.attention.append(Attention(2))
        self.attention.append(Attention(1))

        self.forward1 = nn.ModuleList()
        self.forward1.append(nn.Conv3d(1, 32, kernel_size=3, padding=1))  # 120, 120
        self.forward1.append(enc_block([ 32,  32,  64], (0.5, 0.5, 0.5))) #  60,  60
        self.forward1.append(enc_block([ 64,  64, 128], (0.5, 0.5, 0.5))) #  30,  30
        self.forward1.append(enc_block([128, 128, 256], (0.5, 0.5, 0.5))) #  15,  15

        self.forward2 = nn.ModuleList()
        self.forward2.append(dec_block([256, 128, 128], (2, 2))) #  30,  30
        self.forward2.append(dec_block([256,  64,  64], (2, 2))) #  60,  60
        self.forward2.append(dec_block([128,  32,  32], (2, 2))) # 120, 120

        self.forward3 = nn.Sequential(
                dec_block([64, 16, 16], (2.00, 2.00)), # 240, 240
                dec_block([16,  8,  8], (2.50, 2.50)), # 600, 600
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(8, 2, kernel_size=3, padding=1))

    def forward(self, x):
        x = F.pad(x, (0, 0, 0, 0, 1, 0))

        x0 = self.forward1[0](x)
        x1 = self.forward1[1](x0)
        x2 = self.forward1[2](x1)
        x3 = self.forward1[3](x2)

        y2 = self.forward2[0](self.attention[3](x3))
        y2 = torch.cat([self.attention[2](x2), y2], dim=1)

        y1 = self.forward2[1](y2)
        y1 = torch.cat([self.attention[1](x1), y1], dim=1)

        y0 = self.forward2[2](y1)
        y0 = torch.cat([self.attention[0](x0), y0], dim=1)

        y  = self.forward3(y0)

        return y[:, :1], y[:, 1:]
