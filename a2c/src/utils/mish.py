import torch
from torch import nn
from torch.nn import functional as F


class Mish(nn.Module):
    """
        Purpose: Implements the Mish activation function, a self-regularizing non-linearity that is smooth and differentiable.
    """

    def __init__(self):
        super().__init__()

    def mish(self, input):
        return input * torch.tanh(F.softplus(input))

    def forward(self, input):
        return self.mish(input)
