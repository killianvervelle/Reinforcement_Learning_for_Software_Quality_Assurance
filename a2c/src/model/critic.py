from torch import nn


class Critic(nn.Module):
    """
    Purpose: Implements the critic component of an A2C reinforcement learning algorithm.
    The critic evaluates the value of a given state, which helps estimate the advantage for 
    policy updates in the actor-critic framework.

    Arguments:
        state_dim (int): Dimension of the input state space.
        hidden_size (int): Number of units in the hidden layers.
        activation (nn.Module, optional): Activation function for the hidden layers (default: nn.Tanh).

    Output:
        forward(X): Produces a scalar value (torch.Tensor) representing the estimated value 
        of the given input state.
    """

    def __init__(self, state_dim, hidden_size, activation=nn.Tanh):
        super().__init__()
        self.hidden_size = hidden_size
        self.model = nn.Sequential(
            nn.Linear(state_dim, self.hidden_size),
            activation(),
            nn.Dropout(p=0.2),
            nn.Linear(self.hidden_size, self.hidden_size),
            activation(),
            nn.Dropout(p=0.2),
            nn.Linear(self.hidden_size, 1),
        )

    def forward(self, X):
        return self.model(X)
