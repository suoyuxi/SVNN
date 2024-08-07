import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import sys 

class Mish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh( F.softplus(x) )

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=5, padding=2):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(mid_channels),
            Mish(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            Mish()
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose1d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        '''BCL'''
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class lstm(nn.Module):
    
    def __init__(self, n_features, hidden_size=1):
        super(lstm, self).__init__()
        
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(n_features, n_features, hidden_size)
        
    def forward(self, x):

        batch_size = x.size(0)
        x = x.permute(2, 0, 1) # B*C*L -> L*B*C

        h0 = Variable( torch.zeros(self.hidden_size, batch_size, self.n_features) ).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        c0 = Variable( torch.zeros(self.hidden_size, batch_size, self.n_features) ).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        x, (hn, cn) = self.lstm(x, (h0, c0))
        x = x.permute(1, 2, 0) # L*B*C -> B*C*L

        return x
        
class FilterGenerator(nn.Module):

    def __init__(self, filter_length=15):
        super().__init__()

        self.inc = DoubleConv(1, 16)

        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)

        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)
        self.up4 = Up(32, 16)

        self.outc = OutConv(16,filter_length)

    def forward(self, x):
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        y4 = self.up1(x5, x4)
        y3 = self.up2(y4, x3)
        y2 = self.up3(y3, x2)
        y1 = self.up4(y2, x1)

        f = self.outc(y1)

        return f

class WeightGenerator(nn.Module):
    
    def __init__(self, filter_length=15):
        super().__init__()

        self.reshape_net = nn.Sequential(
            nn.Conv1d(1, filter_length, kernel_size=filter_length, padding=filter_length//2),
            # lstm(filter_length),
            nn.BatchNorm1d(filter_length),
            Mish(),
            nn.Conv1d(filter_length, filter_length, kernel_size=filter_length, padding=filter_length//2),
            # lstm(filter_length),
            nn.BatchNorm1d(filter_length),
            Mish(),
        )

    def forward(self, x):

        x = self.reshape_net(x)

        return x

class svnn(nn.Module):
    
    def __init__(self, filter_length=15):
        super(svnn, self).__init__()

        self.filterGenerator = FilterGenerator(filter_length)
        self.weightGenerator = WeightGenerator(filter_length)
        
        self.outConv = OutConv(filter_length, out_channels=1)

    def forward(self, x):

        sv_filters = self.filterGenerator(x)
        weighted_feature_maps = self.weightGenerator(x)

        filtered_result = sv_filters * weighted_feature_maps

        out_result = self.outConv(filtered_result)

        return F.normalize(out_result, p=float('inf'), dim=2, eps=1e-12, out=None)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = svnn(filter_length=15).to(device=device)
    data_in = torch.randn((64,1,512)).to(device=device)

    data_out = model(data_in)
    print(data_out.size())