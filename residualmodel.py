from torch import nn
from torch.nn import Upsample


class ResidualModel(nn.Module):
    def __init__(self, num_blocks=10, kernel_size=3, num_intermediate_channels=64):
        super().__init__()
        self.residual_func = Upsample(scale_factor=2, mode='bilinear')
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(3, num_intermediate_channels, kernel_size, padding=1),
            *[
                nn.Sequential(
                    nn.Conv2d(num_intermediate_channels,
                              num_intermediate_channels, kernel_size, padding=1),
                    nn.LeakyReLU(),
                ) for _ in range(num_blocks)],
            nn.Conv2d(num_intermediate_channels, 3, kernel_size, padding=1)
        )

    def forward(self, x):
        res = self.layers(x)
        res += self.residual_func(x)
        return res
