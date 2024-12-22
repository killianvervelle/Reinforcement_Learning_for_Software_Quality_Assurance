import gym

import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter

from model.a2cLearner import A2CLearner
from model.actor import Actor
from model.critic import Critic
from model.runner import Runner

from utils.utilities import Utilities
from utils.mish import Mish

from gym.envs import register


class Simulate:

    def __init__(self):
        self.utils = Utilities()
        self.mish = Mish
        self.hidden_size = 128
        self.steps_on_memory = 15
        self.episodes = 70
        self.episode_length = 20

        register(
            id="Env-v0",
            entry_point="my_gym.myGym:Environment",
            kwargs={"render_mode": "rgb_array"}
        )

    def main(self):
        requirement_res_times = [3000, 3050, 3100, 3100]
        sensitivity_collection = [
            [0.79, 0.19, 0.01],
            [0.74, 0.24, 0.01],
            [0.27, 0.63, 0.10],
            [0.42, 0.50, 0.07]]

        vms = self.utils.initialize_vms(
            3, 20, 20, 100, requirement_res_times, sensitivity_collection)
        print(f"Initialized {len(vms)} VMs.")

        # Running the stress test on the first VM.
        vm = vms[0]

        env = gym.make("Env-v0", vm=vm, render_mode="rgb_array")

        writer = SummaryWriter("runs/mish_activation")

        # Config
        state_dim = len(env.observation_space.sample())
        n_actions = env.action_space.n

        actor = Actor(state_dim, n_actions, self.hidden_size,
                      activation=self.mish)
        actor.apply(self.utils.init_xavier_weights)

        critic = Critic(state_dim, self.hidden_size,
                        activation=self.mish)
        critic.apply(self.utils.init_xavier_weights)

        learner = A2CLearner(writer, self.utils, actor, critic)
        runner = Runner(writer, self.utils, env, actor)

        total_steps = (self.episode_length*self.episodes)//self.steps_on_memory

        all_episode_rewards = []
        all_actor_losses = []
        all_critic_losses = []

        for _ in range(total_steps):
            memory, rewards = runner.run(self.steps_on_memory)
            act_loss, crit_loss = learner.learn(
                memory, runner.steps, discount_rewards=True)

            all_episode_rewards.append(rewards)
            all_actor_losses.append(act_loss)
            all_critic_losses.append(crit_loss)

        plt.figure(figsize=(15, 5))

        # Plot episode rewards
        plt.subplot(1, 2, 1)
        plt.plot(all_episode_rewards, label="Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Episode Rewards Over Time")
        plt.legend()
        plt.grid()

        # Plot losses
        plt.subplot(1, 2, 2)
        plt.plot(all_actor_losses, label="Actor Loss")
        plt.plot(all_critic_losses, label="Critic Loss")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title("Actor and Critic Losses Over Time")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    simulator = Simulate()
    simulator.main()
