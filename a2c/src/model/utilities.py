import numpy as np

import torch


class Utilities:
    def __init__(self):
        pass

    # Method to convert a numpy array to a tensor
    def arr_to_tensor(self, input):
        tensor = torch.tensor(input)
        return tensor

utilities = Utilities()