# =============================================================================
# Import required libraries
# =============================================================================
import torch
import torch.nn as nn


class ResidualConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 is_res=False):
        super(ResidualConvBlock, self).__init__()

        self.same_channels = in_channels == out_channels
        self.is_res = is_res

        self.conv1 = nn.Sequential(
            # 3x3 kernel with stride 1 and padding 1
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            #
            if self.same_channels:
                out = x + x2
            else:
                shortcut = nn.Conv2d(x.shape[1],
                                     x2.shape[1],
                                     kernel_size=1,
                                     stride=1,
                                     padding=0).to(x.device)
                out = shortcut(x) + x2
            # normalize output tensor
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(UnetDown, self).__init__()

        self.model = nn.Sequential(
            ResidualConvBlock(in_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
            nn.MaxPool2d(2),  # MaxPool2d layer for downsampling
        )

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(UnetUp, self).__init__()

        self.model = nn.Sequential(
            # ConvTranspose2d layer for upsampling
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        )

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self,
                 input_dim,
                 emb_dim):
        super(EmbedFC, self).__init__()
        '''
        This class defines a generic one layer feed-forward neural network 
        for embedding input data of dimensionality input_dim 
        to an embedding space of dimensionality emb_dim.
        '''
        self.input_dim = input_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x):
        # flatten the input tensor
        x = x.view(-1, self.input_dim)
        return self.model(x)


# =============================================================================
# Sprite U-net
# =============================================================================
class ContextUnetSprite(nn.Module):
    '''
        x dim: (batch-size, 3, 16, 16)
        t dim: (batch-size)
        c dim: (batch-size, 5)
        
        out dim: (batch-size, 3, 16, 16)
    '''

    def __init__(self,
                 in_channels,  # number of input channels
                 n_feat,  # number of intermediate feature maps
                 n_cfeat):  # number of context features (classes)
        super(ContextUnetSprite, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(4), nn.GELU())

        # embed the timestep and context classes with a one-layer
        # fully connected neural network
        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 2*n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 1*n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 4, 4),
            nn.GroupNorm(8, 2 * n_feat),  # normalize
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)

        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, in_channels, 3, 1, 1),
        )

    def forward(self, x, t, c=None):
        # (batch-size, n_feat, 16, 16)
        x = self.init_conv(x)
        # (batch-size, n_feat, 8, 8)
        down1 = self.down1(x)
        # (batch-size, n_feat * 2, 4, 4)
        down2 = self.down2(down1)
        # (batch-size, n_feat * 2, 1, 1)
        hidden_vec = self.to_vec(down2)

        # mask out context if context_mask == 1
        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x.device)

        # embed context and timestep
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # (batch-size, n_feat * 2, 4, 4)
        up_0 = self.up0(hidden_vec)
        # (batch-size, n_feat, 8, 8)
        up_1 = self.up1(cemb1*up_0 + temb1, down2)
        # (batch-size, n_feat, 16, 16)
        up_2 = self.up2(cemb2*up_1 + temb2, down1)
        out = self.out(torch.cat((up_2, x), 1))
        return out
