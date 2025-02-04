from torch import nn
import torch.nn.functional as F


class Network(nn.Module):
    """
    Purpose:
        TThis class represents a neural network backbone designed for reinforcement learning agents, specifically for PPO 
        (Proximal Policy Optimization) and A2C (Advantage Actor-Critic) algorithms, designed to process input features
        and produce either policy predictions (for the actor) or value estimations (for the critic). The network includes
        multiple layers, dropout for regularization, and an activation function for non-linearity. It also includes several 

    Arguments:
        in_features (int): The number of input features to the network.
        hidden_dimensions (int): The number of neurons in the hidden layers.
        out_features (int): The number of output features (e.g., action probabilities for the actor, value estimate for the critic).
        dropout (float): Dropout probability for regularization.
        activation (callable, optional): Activation function applied after each layer. Default is ReLU.

    Notes:
        - Xavier Initialization: The network weights are initialized using Xavier uniform initialization to ensure stable gradients.
        - Gradient Clipping: Gradients are clipped during training to prevent explosion or vanishing gradients.
        - Scheduler: A learning rate scheduler adjusts the learning rate dynamically during training for better convergence.
        - Entropy Coefficient: This regularization encourages exploration by preventing the policy from becoming deterministic too early.

    Output:
        Tensor: The processed output tensor from the final layer of the network.
    """

    def __init__(self, in_features, hidden_dimensions, out_features, dropout, activation=F.relu):
        super().__init__()
        self.layer1 = nn.Linear(in_features, hidden_dimensions)
        self.layer2 = nn.Linear(hidden_dimensions, hidden_dimensions)
        self.layer3 = nn.Linear(hidden_dimensions, out_features)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer3(x)
        return x


class ActorCriticModel(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        return action_pred, value_pred
