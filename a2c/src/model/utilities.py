import numpy as np

import torch


class Utilities:
    def __init__(self):
        pass

    # Method to convert a numpy array to a tensor
    def arr_to_tensor(self, input: np.ndarray):
        tensor = torch.stack(tuple(input), 0)
        return tensor

utilities = Utilities()