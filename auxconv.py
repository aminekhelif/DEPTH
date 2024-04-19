#imports
import torch.nn as nn
class AuxConv(nn.Module):
    def __init__(self, in_channels, c_tag, p=0, downsample=False):
        super(AuxConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, c_tag, kernel_size=(3, 1), stride=1),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Conv2d(c_tag, c_tag, kernel_size=(1, 3), stride=1),
            nn.ReLU(),
            nn.Dropout(p)
        )
        if downsample:
            self.block.add_module('downsample', nn.Conv2d(c_tag, c_tag, kernel_size=3, stride=2))

    def forward(self, x):
        return self.block(x)
