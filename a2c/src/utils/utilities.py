import torch
from torch import nn


class Utilities:
    """
    Purpose:
        This class contains utility functions used during the training of reinforcement learning models. 
        These utilities include methods for initializing network weights, clipping gradients, and normalizing rewards. 
        They help stabilize and improve the convergence of the learning process.

    Notes:
        - Xavier Initialization: Ensures stable gradients by using Xavier uniform initialization for the network weights.
        - Gradient Clipping: Helps prevent exploding or vanishing gradients during backpropagation by clipping gradients.
        - Scheduler: A learning rate scheduler dynamically adjusts the learning rate to facilitate better training convergence.
        - Entropy Coefficient: Used as regularization to encourage exploration, preventing the policy from becoming deterministic too early.

    Methods:
        - init_xavier_weights: Applies Xavier uniform initialization to the weights of linear layers.
        - clip_grad_norm_: Clips gradients during backpropagation to prevent exploding gradients.
        - clip_logits: Clips logits to a specific range. 
        - normalize_rewards: Normalizes a batch of rewards by subtracting the mean and dividing by the standard deviation.
    """

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
