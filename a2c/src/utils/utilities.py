import torch
from torch import nn


class Utilities:

    def __init__(self):
        pass

    @staticmethod
    def init_xavier_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def clip_grad_norm_(module, max_grad_norm):
        params = [p for g in module.param_groups for p in g["params"]
                  if p.grad is not None]
        """print("Gradients before clipping:")
        for i, p in enumerate(params):
            print(f"Parameter {i}: Norm = {p.grad.norm():.6f}")"""
        total_norm = nn.utils.clip_grad_norm_(params, max_grad_norm)
        """print("\nGradients after clipping:")
        for i, p in enumerate(params):
            print(f"Parameter {i}: Norm = {p.grad.norm():.6f}")"""

    @staticmethod
    def clip_logits(logits, clip_value):
        return torch.clamp(logits, -clip_value, clip_value)

    @staticmethod
    def normalize_rewards(r):
        mean = r.mean()
        std = r.std() + 1e-8
        return (r - mean) / std
