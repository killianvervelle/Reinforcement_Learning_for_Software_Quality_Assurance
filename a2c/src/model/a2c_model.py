import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Categorical

from utilities import utilities


class a2c_model(nn.Module):

    def __init__(self, env, hidden_size: int, gamma: float, initial_state:np.array, random_seed=None) -> None:
        super.__init__()

        if random_seed:
            torch.manual_seed(random_seed)

        self.env = env
        self.gamma = gamma
        self.hidden_size = hidden_size
        self.in_size = len(self.env.observation_space.sample().flatten())
        self.out_size = len(self.env.action_space)
        self.initial_state = initial_state

        self.actor = nn.Sequential(
            nn.Linear(self.in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.out_size)
        )

        self.critic = nn.Sequential(
            nn.Linear(self.in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def train_model(self, logging=False):
        """
        Trains the agent by adjusting its policy.
        
        Returns: 
            tuple: rewards, tensor(critic_values), tensor(actor_probs), total_reward
        """
        rewards = []
        critic_values = []
        actor_probs = []

        observation = self.initial_state

        done = False
        while not done:
            observation = torch.from_numpy(observation)
            # Get action from Actor
            action_logits = self.actor(observation)
            action = Categorical(logits=action_logits).sample()
            action_probs = action_logits[action]
            actor_probs.append(action_probs)

            # Get predicted value from the Critic
            pred = torch.squeeze(self.critic(observation).view(-1))
            critic_values.append(pred)

            # Run action in environment and get rewards, next state
            observation, reward, done = self.env.step(action.item())
            rewards.append(torch.tensor(reward))

        total_reward = sum(rewards)
        
        for eps in range(len(total_reward)):
            cum_reward = 0
            for step in range(eps, len(reward)):
                # Discounted futur reward for each episode
                cum_reward += reward[step] * (self.gamma ** (step-eps))
            reward[eps] = cum_reward

        # Convert output arrays to tensors
        rewards = utilities.arr_to_tensor(rewards)
        # Zero-mean standardization of rewards
        rewards = (rewards - torch.mean(rewards)) / (torch.std(rewards) + .0000000001)

        return rewards, utilities.arr_to_tensor(critic_values), utilities.arr_to_tensor(actor_probs), total_reward
    
    def test_model(self):
        """
        Evaluates the agent's performance on the learned policy.

        Returns: 
            float: Total reward accumulated during the test episode.
        """
        rewards = []
        done = False
        while not done:
            observation = torch.from_numpy(observation)

            # Get action from Actor
            action_logits = self.actor(observation)
            action = Categorical(logits=action_logits).sample()
            
            # Exploit the learned policy, get rewards and new states
            observation, reward, done = self.env.step(action.item())
            rewards.append(reward)

        return sum(rewards)
    

    @staticmethod
    def compute_loss(
        actor_probs: torch.Tensor,
        actual_returns: torch.Tensor, 
        expected_returns: torch.Tensor,
        critic_loss_fn=nn.SmoothL1Loss()
        ) -> torch.Tensor:
        """
        Computes the combined loss for Actor-Critic.

        Args:
            actor_probs (torch.Tensor): Log probabilities of actions from the actor.
            actual_returns (torch.Tensor): The computed returns (target values).
            expected_returns (torch.Tensor): The critic's predicted values for returns.
            critic_loss_fn (callable): Loss function to compute critic loss. Default is SmoothL1Loss.

        Returns:
            torch.Tensor: Combined loss (actor loss + critic loss).
        """
        assert len(actor_probs) == len(actual_returns) == len(expected_returns)

        advantage = actual_returns - expected_returns # .detach()

        return -(torch.sum(actor_probs * advantage), critic_loss_fn(actual_returns, expected_returns))