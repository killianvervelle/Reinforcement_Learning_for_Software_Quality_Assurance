import torch
from torch import nn


class Actor(nn.Module):
    """
    Purpose: Implements the actor component of an A2C reinforcement learning algorithm.
    The actor generates an action distribution given an input state allowing the agent to 
    select actions based on this distribution.

    Arguments:
        state_dim (int): Dimension of the input state space.
        n_actions (int): Number of possible actions the agent can take.
        hidden_size (int): Number of units in the hidden layers.
        activation (nn.Module, optional): Activation function for the hidden layers (default: nn.Tanh).

    Output:
        forward(X): Produces a torch.distributions.Normal object with mean and standard deviation
        representing the action distribution for the given input state.
    """

    def __init__(self, state_dim, n_actions, hidden_size, activation=nn.Tanh):
        super().__init__()
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.model = nn.Sequential(
            nn.Linear(state_dim, self.hidden_size),
            activation(),
            nn.Dropout(p=0.2),
            nn.Linear(self.hidden_size, self.hidden_size),
            activation(),
            nn.Dropout(p=0.2),
            nn.Linear(self.hidden_size, n_actions)
        )

        logstds_param = nn.Parameter(torch.full((n_actions,), 0.01))
        self.register_parameter("logstds", logstds_param)

    def forward(self, X):
        means = self.model(X)
        stds = torch.clamp(self.logstds.exp(), 1e-3, 50)

        return torch.distributions.Normal(means, stds)
