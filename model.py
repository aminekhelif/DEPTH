import torch
import torch.nn as nn
from torchvision.models import resnet152

class ModifiedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super(ModifiedBlock, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1).to(self.device)
        self.bn1 = nn.BatchNorm2d(in_channels // 4).to(self.device)
        self.relu = nn.ReLU(inplace=True).to(self.device)
        self.conv2 = nn.Conv2d(in_channels // 4, in_channels // 4, 3, padding=1).to(self.device)
        self.bn2 = nn.BatchNorm2d(in_channels // 4).to(self.device)
        self.conv3 = nn.Conv2d(in_channels // 4, out_channels, 1).to(self.device)
        self.bn3 = nn.BatchNorm2d(out_channels).to(self.device)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x += residual
        x = self.relu(x)
        return x

class DepthEstimationNet(nn.Module):
    def __init__(self, num_aux_modules=16, aux_channels=16, device='cpu'):
        super(DepthEstimationNet, self).__init__()
        self.device = device
        base_resnet = resnet152(pretrained=True).to(self.device)
        self.features = nn.Sequential(*list(base_resnet.children())[:-2]).to(self.device)

        # Modified Blocks
        self.modified_blocks = nn.ModuleList(
            [ModifiedBlock(1024, 1024, self.device) for _ in range(num_aux_modules // 2)] +
            [ModifiedBlock(2048, 2048, self.device) for _ in range(num_aux_modules // 2)]
        ).to(self.device)

        # Auxiliary Convolutions
        self.aux_convs = nn.ModuleList(
            [self._create_aux_module(1024, aux_channels) for _ in range(num_aux_modules // 2)] +
            [self._create_aux_module(2048, aux_channels) for _ in range(num_aux_modules // 2)]
        ).to(self.device)

        # Depth Estimation Head
        self.depth_head = nn.Sequential(
            nn.Conv2d(aux_channels * num_aux_modules, 128, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 3, stride=2, padding=1, output_padding=1)
        ).to(self.device)

    def _create_aux_module(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1).to(self.device),
            nn.BatchNorm2d(out_channels).to(self.device),
            nn.ReLU(inplace=True).to(self.device)
        )

    def forward(self, x):
        x = x.to(self.device)
        x = self.features(x)

        aux_features = []
        for mod_block, aux_conv in zip(self.modified_blocks, self.aux_convs):
            x = mod_block(x)
            aux = aux_conv(x)
            aux_features.append(aux)

        x = torch.cat(aux_features, dim=1)
        depth_map = self.depth_head(x)
        return depth_map

if __name__ == "__main__":
    # Set the device, compatible with MPS for Apple silicon if available
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Example usage
    model = DepthEstimationNet(device=device)
    input_tensor = torch.randn(1, 3, 224, 224)  # Example input
    output = model(input_tensor.to(device))
    print(output.size())  # Should be the expected output size
