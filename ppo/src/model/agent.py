import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from model.backboneNetwork import BackboneNetwork
from model.backboneNetwork import ActorCritic

from utils.utilities import Utilities


class Agent:
    """
    Purpose:
        This class represents a reinforcement learning agent capable of training and evaluating on a specified environment. 
        It implements an actor-critic method with PPO optimizations and supports training with reward calculation, 
        policy update, and evaluation.

    Methods:
        create_agent: Builds an Actor-Critic agent with configurable hidden dimensions and dropout.
        calculate_returns: Computes the discounted returns for the agent using rewards from the environment.
        calculate_advantages: Calculates the advantages based on the difference between returns and value predictions.
        calculate_surrogate_loss: Computes the surrogate loss for PPO based on the old and new log probabilities.
        calculate_ppo_losses: Computes policy and value losses using the surrogate loss and entropy bonus.
        init_training: Initializes and returns empty lists for training-related variables.
        forward_pass: Runs a forward pass through the environment, collects states, actions, and rewards, and computes returns and advantages.
        update_policy: Updates the agent's policy using PPO losses by performing optimization steps.
        evaluate: Evaluates the agent's performance on a given environment and returns the total reward for the episode.
        plot_train_rewards: Plots the training rewards over the course of episodes and compares with the reward threshold.
        plot_test_rewards: Plots the testing rewards over the course of episodes and compares with the reward threshold.
        plot_losses: Plots both policy and value losses during training to visualize convergence.
        run_agent: Executes the full training and evaluation cycle, including policy updates, evaluation, and reward tracking.

    Attributes:
        env_train: The training environment where the agent interacts and learns.
        env_test: The testing environment used to evaluate the agent's performance during training.
        utilities: A utility class providing helper functions like weight initialization and gradient clipping.
    """

    def __init__(self, env_train, env_test):
        self.env_train = env_train
        self.env_test = env_test

    def create_agent(self, hidden_dimensions, dropout):
        INPUT_FEATURES = self.env_train.observation_space.shape[0]
        HIDDEN_DIMENSIONS = hidden_dimensions
        ACTOR_OUTPUT_FEATURES = self.env_train.action_space.n
        CRITIC_OUTPUT_FEATURES = 1
        DROPOUT = dropout

        actor = BackboneNetwork(
            INPUT_FEATURES, HIDDEN_DIMENSIONS, ACTOR_OUTPUT_FEATURES, DROPOUT)
        actor.apply(Utilities.init_xavier_weights)

        critic = BackboneNetwork(
            INPUT_FEATURES, HIDDEN_DIMENSIONS, CRITIC_OUTPUT_FEATURES, DROPOUT)
        critic.apply(Utilities.init_xavier_weights)

        agent = ActorCritic(actor, critic)
        return agent

    def calculate_returns(self, rewards, discount_factor):
        returns = []
        cumulative_reward = 0
        for r in reversed(rewards):
            cumulative_reward = r + cumulative_reward * discount_factor
            returns.insert(0, cumulative_reward)
        returns = torch.tensor(returns)
        return Utilities.normalize_rewards(returns)

    def calculate_advantages(self, returns, values):
        advantages = returns - values
        normalized_advantages = (
            advantages - advantages.mean()) / advantages.std()
        return normalized_advantages

    def calculate_surrogate_loss(
            self,
            actions_log_probability_old,
            actions_log_probability_new,
            epsilon,
            advantages):
        advantages = advantages.detach()
        policy_ratio = (
            actions_log_probability_new - actions_log_probability_old
        ).exp()
        surrogate_loss_1 = policy_ratio * advantages
        surrogate_loss_2 = torch.clamp(
            policy_ratio, min=1.0-epsilon, max=1.0+epsilon
        ) * advantages
        surrogate_loss = torch.min(surrogate_loss_1, surrogate_loss_2)
        return surrogate_loss

    def calculate_ppo_losses(
            self,
            surrogate_loss,
            entropy,
            entropy_coefficient,
            returns,
            value_pred):
        entropy_bonus = entropy_coefficient * entropy
        policy_loss = -(surrogate_loss + entropy_bonus).sum()
        value_loss = F.smooth_l1_loss(returns, value_pred).sum()
        return policy_loss, value_loss

    def init_training(self):
        states = []
        actions = []
        actions_log_probability = []
        values = []
        rewards = []
        done = False
        episode_reward = 0
        return states, actions, actions_log_probability, values, rewards, done, episode_reward

    def forward_pass(self, env, agent, discount_factor):
        states, actions, actions_log_probability, values, rewards, done, episode_reward = self.init_training()
        state = env.reset()
        agent.train()
        while not done:
            state = state[0] if isinstance(state, tuple) else state
            state = torch.FloatTensor(state).unsqueeze(0).view(1, -1)
            states.append(state)
            action_pred, value_pred = agent(state)
            action_prob = F.softmax(action_pred, dim=-1)
            dist = torch.distributions.Categorical(action_prob)
            action = dist.sample()
            log_prob_action = dist.log_prob(action)
            state, reward, done, _, _ = env.step(action.item())
            actions.append(action)
            actions_log_probability.append(log_prob_action)
            values.append(value_pred)
            rewards.append(reward)
            episode_reward += reward
        states = torch.cat(states)
        actions = torch.cat(actions)
        actions_log_probability = torch.cat(actions_log_probability)
        values = torch.cat(values).squeeze(-1)
        returns = self.calculate_returns(rewards, discount_factor)
        advantages = self.calculate_advantages(returns, values)
        return episode_reward, states, actions, actions_log_probability, advantages, returns

    def update_policy(
            self,
            agent,
            states,
            actions,
            actions_log_probability_old,
            advantages,
            returns,
            optimizer,
            scheduler,
            max_grad_norm,
            ppo_steps,
            epsilon,
            entropy_coefficient):
        BATCH_SIZE = 128
        total_policy_loss = 0
        total_value_loss = 0
        actions_log_probability_old = actions_log_probability_old.detach()
        actions = actions.detach()
        training_results_dataset = TensorDataset(
            states,
            actions,
            actions_log_probability_old,
            advantages,
            returns)
        batch_dataset = DataLoader(
            training_results_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False)
        for _ in range(ppo_steps):
            for _, (states, actions, actions_log_probability_old, advantages, returns) in enumerate(batch_dataset):
                action_pred, value_pred = agent(states)
                value_pred = value_pred.squeeze(-1)
                action_prob = F.softmax(action_pred, dim=-1)
                probability_distribution_new = torch.distributions.Categorical(
                    action_prob)
                entropy = probability_distribution_new.entropy()
                actions_log_probability_new = probability_distribution_new.log_prob(
                    actions)
                surrogate_loss = self.calculate_surrogate_loss(
                    actions_log_probability_old,
                    actions_log_probability_new,
                    epsilon,
                    advantages)
                policy_loss, value_loss = self.calculate_ppo_losses(
                    surrogate_loss,
                    entropy,
                    entropy_coefficient,
                    returns,
                    value_pred)
                optimizer.zero_grad()
                policy_loss.backward()
                value_loss.backward()
                # Clipping gradient to avoid explosion or vanishing gradient
                Utilities.clip_grad_norm_(optimizer, max_grad_norm)
                optimizer.step()
                # Scheduler to dynamically adjust the learning rate over training
                scheduler.step()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
        return total_policy_loss / ppo_steps, total_value_loss / ppo_steps

    def evaluate(self, env, agent):
        agent.eval()
        rewards = []
        done = False
        episode_reward = 0
        state = env.reset()
        while not done:
            state = state[0] if isinstance(state, tuple) else state
            state = torch.FloatTensor(state).unsqueeze(0).view(1, -1)
            with torch.no_grad():
                action_pred, _ = agent(state)
                action_prob = F.softmax(action_pred, dim=-1)
            action = torch.argmax(action_prob, dim=-1)
            state, reward, done, _, _ = env.step(action.item())
            rewards.append(reward)
            episode_reward += reward
        return episode_reward

    def plot_train_rewards(self, train_rewards, reward_threshold):
        plt.figure(figsize=(12, 8))
        plt.plot(train_rewards, label='Training Reward')
        plt.xlabel('Episode', fontsize=20)
        plt.ylabel('Training Reward', fontsize=20)
        plt.hlines(reward_threshold, 0, len(train_rewards), color='y')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

    def plot_test_rewards(self, test_rewards, reward_threshold):
        plt.figure(figsize=(12, 8))
        plt.plot(test_rewards, label='Testing Reward')
        plt.xlabel('Episode', fontsize=20)
        plt.ylabel('Testing Reward', fontsize=20)
        plt.hlines(reward_threshold, 0, len(test_rewards), color='y')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

    def plot_losses(self, policy_losses, value_losses):
        plt.figure(figsize=(8, 4))
        plt.plot(value_losses, label='Value Losses')
        plt.plot(policy_losses, label='Policy Losses')
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

    def run_agent(self):
        MAX_EPISODES = 300
        # A lower 𝛾 places more weight on immediate rewards and less weight on future rewards.
        DISCOUNT_FACTOR = 0.99
        REWARD_THRESHOLD = 12
        MIN_REWARD_THRESHOLD = -10
        PRINT_INTERVAL = 10
        PPO_STEPS = 8
        N_TRIALS = 30
        EPSILON = 0.2
        ENTROPY_COEFFICIENT = 1000
        HIDDEN_DIMENSIONS = 64
        DROPOUT = 0.2
        LEARNING_RATE = 0.001
        MAX_GRAD_NORM = 100.0
        train_rewards = []
        test_rewards = []
        policy_losses = []
        value_losses = []
        agent = self.create_agent(HIDDEN_DIMENSIONS, DROPOUT)
        optimizer = torch.optim.Adam(agent.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=100, gamma=0.9)
        for episode in range(1, MAX_EPISODES+1):
            train_reward, states, actions, actions_log_probability, advantages, returns = self.forward_pass(
                self.env_train,
                agent,
                DISCOUNT_FACTOR)
            policy_loss, value_loss = self.update_policy(
                agent,
                states,
                actions,
                actions_log_probability,
                advantages,
                returns,
                optimizer,
                scheduler,
                MAX_GRAD_NORM,
                PPO_STEPS,
                EPSILON,
                ENTROPY_COEFFICIENT)
            test_reward = self.evaluate(self.env_test, agent)
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            train_rewards.append(train_reward)
            test_rewards.append(test_reward)
            mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
            mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
            mean_abs_policy_loss = np.mean(np.abs(policy_losses[-N_TRIALS:]))
            mean_abs_value_loss = np.mean(np.abs(value_losses[-N_TRIALS:]))
            if episode % PRINT_INTERVAL == 0:
                print(f'Episode: {episode:3} | \
                        Mean Train Rewards: {mean_train_rewards:3.1f} \
                        | Mean Test Rewards: {mean_test_rewards:3.1f} \
                        | Mean Abs Policy Loss: {mean_abs_policy_loss:2.2f} \
                        | Mean Abs Value Loss: {mean_abs_value_loss:2.2f}')
            if mean_test_rewards >= REWARD_THRESHOLD or mean_test_rewards <= MIN_REWARD_THRESHOLD:
                print(f'Reached reward threshold in {episode} episodes')
                break
        self.plot_train_rewards(train_rewards, REWARD_THRESHOLD)
        self.plot_test_rewards(test_rewards, REWARD_THRESHOLD)
        self.plot_losses(policy_losses, value_losses)