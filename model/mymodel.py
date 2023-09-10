import torch
from torch import nn
import torch.nn.functional as F

class GConv(nn.Module):
    def __init__(self, input_shape, output_nc, group_num=4):
        super(GConv, self).__init__()

        input_nc = input_shape[0]
        h = input_shape[1]
        w = input_shape[2]

        self.layer = nn.Conv2d(input_nc, output_nc, groups=group_num, kernel_size=(h, w), stride=(h, w), padding=0)

    def forward(self, x):
        out = self.layer(x)     # B, OC, 1
        return out.squeeze()    # B, OC

class MixedDilationConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        dil1 =  out_channels // 2
        dil2 = out_channels - dil1
        '''
        dim = Din + 2*pad - dilation*(k-1)
        if dim == Din:
            pad = dilation(k-1) // 2
        '''
        padding1 = ((kernel_size[0]-1)//2, (kernel_size[1]-1)//2)
        self.conv1 = nn.Conv2d(in_channels, dil1, kernel_size=kernel_size, padding=padding1, dilation=1)
        padding2 = ((kernel_size[0]-1), (kernel_size[1]-1))
        self.conv2 = nn.Conv2d(in_channels, dil2, kernel_size=kernel_size, padding=padding2, dilation=2)

    def forward(self, x):
        o1 = self.conv1(x)  # [B, c1, H, W]
        o2 = self.conv2(x)  # [B, c2, H, W]
        out = torch.cat((o1, o2), dim=1)  # [B, out_channels, H, W]

        return out  

class EncoderModel(nn.Module):
    def __init__(self, layer_size, hidden_nc, kernel_size, feature_dim, dropout=0.5):
        super(EncoderModel, self).__init__()

        H = 148
        W = 7
        layers = []
        for layer in range(layer_size):
            if layer == 0:
                input_cn = 2
            else:
                input_cn = hidden_nc
            layers.append(MixedDilationConv(input_cn, hidden_nc, kernel_size))
            layers.append(nn.LayerNorm((hidden_nc, H, W)))
            layers.append(nn.PReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(GConv((hidden_nc, H, W), feature_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)
        

class MYModel(nn.Module):
    def __init__(self, encoder_layer_size, hidden_nc, kernel_size, feature_dim, project_dim, classify_dim):
        super(MYModel, self).__init__()

        self.encoder = EncoderModel(encoder_layer_size, hidden_nc, kernel_size, feature_dim)
        self.projtnet = nn.Linear(feature_dim, project_dim)

        self.classifier = nn.Sequential( 
            nn.Linear(feature_dim, classify_dim),
            nn.BatchNorm1d(classify_dim),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(classify_dim, 2)
        )
          
    def forward(self, x, xx=None):
        x_encode = self.encoder(x)                  # [B, F]          
        x_classify = self.classifier(x_encode)      # [B, 2]
        if self.training:
            xx_encode = self.encoder(xx)            # [2B, F]
            xx_project = F.normalize(self.projtnet(xx_encode), dim=1)   # [2B, PF]
            return x_classify, xx_project
        else:
            return x_classify