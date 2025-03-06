import torch
from torch import nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self): super().__init__()

    def mish(self, input): return input * torch.tanh(F.softplus(input))

    def forward(self, input): return self.mish(input)
