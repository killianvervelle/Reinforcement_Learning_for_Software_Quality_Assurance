import torch
from torch.distributions import Categorical


class Runner():
    """
    Purpose: Defines an environment runner that interacts with the environment, collects 
    experiences, and stores them in memory. It tracks the state, actions, rewards, and episodes.

    Arguments:
        env (gym.Env): The environment instance that the agent interacts with.

    Output:
        None: Initializes the Runner with an environment and necessary attributes.
    """

    def __init__(self, writer, utils, env, actor):
        self.env = env
        self.actor = actor
        self.state = None
        self.done = True
        self.steps = 0
        self.episode_reward = 0
        self.episode_rewards = []
        self.utils = utils
        self.writer = writer

    def reset(self):
        self.episode_reward = 0
        self.done = False
        self.state = self.env.reset()

    def run(self, max_steps, memory=None):
        """Run the environment for a specified number of steps and store the experience in memory."""
        if memory is None:
            memory = []

        for _ in range(max_steps):
            if self.done:
                self.reset()
            # Extract the state (in case it is a tuple of state and info)
            state = self.state[0] if isinstance(
                self.state, tuple) else self.state

            # Pass the state to the actor
            dists = self.actor(self.utils.to_tensor(state))
            actions = dists.sample().detach().data.numpy()
            actions_tensor = torch.tensor(actions, dtype=torch.float32)
            action = Categorical(logits=actions_tensor).sample()
            # Take a step in the environment
            next_state, reward, self.done, _, _ = self.env.step(action)

            # Store the experience in memory
            memory.append((action, reward, self.state, next_state, self.done))

            # Update the state
            self.state = next_state

            # Track the steps and rewards
            self.steps += 1
            self.episode_reward += reward

            # If the episode is done, append the reward and log it
            if self.done:
                self.episode_rewards.append(self.episode_reward)
                if len(self.episode_rewards) % 10 == 0:
                    print("Episode:", len(self.episode_rewards),
                          ", Episode reward:", self.episode_reward)
                self.writer.add_scalar("episode_reward",
                                       self.episode_reward, global_step=self.steps)

        return memory, self.episode_reward
