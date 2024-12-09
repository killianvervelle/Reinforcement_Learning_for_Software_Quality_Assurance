import numpy as np
import gym
import time
import torch
import torch.nn as nn
from torch.distributions import Categorical


class A2C(nn.Module):

    def __init__(self, hidden_size, vm, gamma, random_seed=None):
        super(A2C, self).__init__()

        if random_seed:
            torch.manual_seed(random_seed)

        env = gym.make("StressingEnv-v0", vm=vm, render_mode="rgb_array")

        self.env = env
        self.gamma = gamma
        self.hidden_size = hidden_size
        self.in_size = len(self.env.observation_space.sample())   
        self.out_size = self.env.action_space.n

        self.actor = nn.Sequential(
            nn.Linear(self.in_size, hidden_size),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_size, self.out_size)
        ).double()

        self.critic = nn.Sequential(
            nn.Linear(self.in_size, hidden_size),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_size, 1)
        ).double()

        # Weight initlization
        self.actor.apply(self.init_xavier_weights)
        self.critic.apply(self.init_xavier_weights)
    
    def train_model(self, render=False):
        """
        Trains the agent by adjusting its policy.
        
        Returns: 
            tuple: rewards, tensor(critic_values), tensor(actor_probs), total_reward
        """
        rewards = []
        critic_values = []
        actor_probs = []

        observation = self.env.reset()
        print("Observation: ", observation)
        
        terminated = False
        while not terminated:
            observation_array = np.array(list(observation.values()), dtype=np.float32)

            if render:
                self.env.render()
    
            observation = torch.from_numpy(observation_array).double()
            print("Training with new observation: ", observation)

            # Get action from Actor
            action_logits = self.actor(observation)

            action_logits = torch.clamp(action_logits, min=-1000, max=1000)

            print("ACTION LOGIT", action_logits)
            action = Categorical(logits=action_logits).sample()
            action_probs = action_logits[action]

            actor_probs.append(action_probs)

            # Get predicted value from the Critic
            pred = torch.squeeze(self.critic(observation).view(-1))
            critic_values.append(pred)

            # Run action in environment and get rewards, next state
            observation, reward, terminated, _, _ = self.env.step(action.item())

            rewards.append(torch.tensor(reward).double())

        total_reward = sum(rewards)

        print("TOTAL REWARD", total_reward)
        
        for eps in range(len(rewards)):
            cum_reward = 0
            for step in range(eps, len(rewards)):
                # Discounted futur reward for each episode
                cum_reward += rewards[step] * (self.gamma ** (step-eps))
            rewards[eps] = cum_reward

        # Convert output arrays to tensors
        rewards = self.arr_to_tensor(rewards)

        # Zero-mean standardization of rewards
        rewards = (rewards - torch.mean(rewards)) / (torch.std(rewards) + .0000000001)

        return rewards, self.arr_to_tensor(critic_values), self.arr_to_tensor(actor_probs), total_reward
    
    def test_model(self, render="rgb_array"):
        """
        Evaluates the agent's performance on the learned policy.

        Returns: 
            float: Total reward accumulated during the test episode.
        """
        rewards = []

        observation = self.env.reset()
        print("Testing with new observation: ", observation)
        
        terminated = False
        while not terminated:
            time.sleep(0.5)

            observation_array = np.array(list(observation.values()), dtype=np.float32)

            observation = torch.from_numpy(observation_array).double()

            # Get action from Actor
            action_logits = self.actor(observation)
            action = Categorical(logits=action_logits).sample()
            
            # Exploit the learned policy, get rewards and new states
            observation, reward, terminated, _, _ = self.env.step(action.item())
            rewards.append(reward)

            if render:
                self.env.render()

        return sum(rewards)
    
    def arr_to_tensor(self, input):
        return torch.stack(tuple(input), 0)
    
    def init_xavier_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias) 
    
    @staticmethod
    def compute_loss(
        actor_probs,
        actual_returns, 
        expected_returns,
        critic_loss_fn=nn.SmoothL1Loss(),
        entropy_weight=0.001
        ):
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

        actual_returns = torch.nan_to_num(actual_returns, nan=0.0)
        expected_returns = torch.nan_to_num(expected_returns, nan=0.0)

        actual_returns = torch.clamp(actual_returns, -1e4, 1e4)
        expected_returns = torch.clamp(expected_returns, -1e4, 1e4)

        
        advantage = actual_returns - expected_returns.detach()

        print("advtange", advantage)

        print("actual return", actual_returns)

        print("expected returns", expected_returns)

        entropy = -torch.sum(actor_probs * torch.log(actor_probs + 1e-8), dim=-1).mean()

        print("ENTROPY", entropy)

        actor_loss = -(torch.sum(actor_probs * advantage)) -  entropy_weight * entropy

        return actor_loss, critic_loss_fn(actual_returns, expected_returns)
    