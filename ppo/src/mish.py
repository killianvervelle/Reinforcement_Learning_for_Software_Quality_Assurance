import torch
from torch import nn
import torch.nn.functional as F


class Mish(nn.Module):
    """
    Purpose:
        This class defines the Mish activation function, a self-regularizing non-linear activation function that
        has shown improved performance in various neural network architectures.

    Notes:
        Mish Activation: The Mish activation function is defined as:
            mish(x) = x * tanh(softplus(x))
        It is known for providing smoother gradients and better performance than ReLU and other traditional activation functions.
    """

    def __init__(self): super().__init__()
    def mish(self, input): return input * torch.tanh(F.softplus(input))
    def forward(self, input): return self.mish(input)
