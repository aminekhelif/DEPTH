import torch
import torch.nn as nn
from torchvision.models import resnet152
from auxconv import *

class DEPTH(nn.Module):
    def __init__(self, wts=None, freeze=True, p=0):
        super(DEPTH, self).__init__()
        self.resnet = resnet152(pretrained=False)
        if wts:
            self.resnet.fc = nn.Linear(2048, 800)  # redefine the fully connected layer
            self.resnet.load_state_dict(torch.load(wts))
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

        self.initialize_network()

        # Prepare auxiliary convolutional modules
        self.aux_modules_1024 = nn.ModuleList([AuxConv(in_channels=1024, c_tag=16, p=p, downsample=True) for _ in range(13)])
        self.aux_modules_2048 = nn.ModuleList([AuxConv(in_channels=2048, c_tag=16, p=p) for _ in range(3)])
        self.aux_modules = nn.ModuleList(list(self.aux_modules_1024) + list(self.aux_modules_2048))

        # Initialize weights for the added auxiliary convolution modules
        self._init_added_weights()

    def initialize_network(self):
        self.resnet_top, self.resnet_mid, self.avg_pool2d, self.deconv = self.flatten_model(self.resnet)


    def flatten_model(self, model):
        if not isinstance(model, nn.Module):
            raise ValueError("The provided model must be a PyTorch model (nn.Module).")

        flattened = []
        children = list(model.children())  # Get a list of all children modules
        flattened += children[:4]

        # Flatten internal structures of blocks from layer 4 to 7
        for i in range(4, 8):
            sequence = children[i]
            flattened += list(sequence.children())  # Ensure children of blocks are expanded

        flattened += children[-2:]  # Append the last layers

        # Create sequences and module lists from the flattened structure
        resnet_top = nn.Sequential(*flattened[:38])
        resnet_mid = nn.ModuleList(flattened[38:54])
        avg_pool2d = flattened[54]
        deconv = nn.Sequential(
            self._deconv_block(256, 3, 2, 1),
            self._deconv_block(64, 3, 2, [2, 1]),
            self._deconv_block(16, 3, 2, [2, 1]),
            self._deconv_block(4, [3, 4], 1, 2)
        )

        return resnet_top, resnet_mid, avg_pool2d, deconv


    def _deconv_block(self, in_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, 1, 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU()
        )

    def _init_added_weights(self):
        for module in self.aux_modules:
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)

    def forward(self, x):
        x = self.resnet_top(x)
        outputs = []
        
        for i, block in enumerate(self.resnet_mid):
            x = block(x)
            
            if i < len(self.aux_modules_1024):
                output = self.aux_modules_1024[i](x)
            else:
                output = self.aux_modules_2048[i - len(self.aux_modules_1024)](x)
            
            print(f"Output shape at index {i}: {output.shape}")  # Debug line to print shape
            outputs.append(output)

        # Before concatenating, ensure all tensors have the same height and width.
        # If this is not the case, you need to resize or pad them to match.
        # Example: outputs = [F.interpolate(o, size=outputs[0].shape[2:]) for o in outputs]
        
        # Verify if all outputs have the same shape except in the channel dimension
        assert all(o.shape[2] == outputs[0].shape[2] and o.shape[3] == outputs[0].shape[3] for o in outputs), "Mismatched feature map sizes."

        x = torch.cat(outputs, dim=1)  # Now we concatenate along the channel dimension.
        x = self.deconv(x)
        x = x.view(x.shape[0], -1)
        return x


