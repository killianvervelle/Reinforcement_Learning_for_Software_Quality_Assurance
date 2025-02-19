import warnings
import numpy as np
import torch
import gym
import logging
import sys
import os
import argparse

from utils.utilities import Utilities
from virtualMachine import VirtualMachine
from optimizer import Optimizer
from gym.envs import register
from agent import Agent
from sampler import Sampler

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

warnings.filterwarnings("ignore", category=DeprecationWarning)

np.bool8 = np.bool_

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ["JVM_ARGS"] = "-Dlog_level.jmeter=OFF"

model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model/")
model_path = os.path.join(model_dir, "ppo_trained_model.pth")


class Environment:
    """
    Purpose:
        This class manages the simulation of virtual machines (VMs) and reinforcement learning environments. 
        It initializes VMs, registers the custom environment, and runs the simulation loop involving an agent.

    Methods:
        initialize_vms: Initializes a set of virtual machines with specific characteristics.
        initialize_env: Creates training and testing environments for reinforcement learning.
        load_model: Loads the model model used to predict the system's response time.
    """

    def __init__(self):
        pass

    def initialize_vms(self, n, requirement_res_times):
        vm_list = []

        for i in range(n):
            vm = VirtualMachine(
                Requirement_ResTime=requirement_res_times[i]
            )
            vm_list.append(vm)

        return vm_list

    def initialize_env(self, model):
        requirement_res_times = [3000, 3000]
        vms = self.initialize_vms(2, requirement_res_times)
        print(f"Initialized {len(vms)} VMs.")

        if "Env-v1" not in gym.envs.registry:
            register(
                id="Env-v1",
                entry_point="my_gym.myGymStress:ResourceStarving"
            )

        env_train = gym.make("Env-v1", vm=vms[0], model=model)
        env_test = gym.make("Env-v1", vm=vms[0], model=model)

        return env_train, env_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--discount_factor", type=float, default=0.90)
    parser.add_argument("--epsilon", type=float, default=0.35)
    parser.add_argument("--entropy_coefficient", type=float, default=0.08)
    parser.add_argument("--hidden_dimensions", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.22)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.0008)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument("--reward_threshold", type=float, default=30.0)
    parser.add_argument("--ppo_steps", type=float, default=12)
    args = parser.parse_args()

    env = Environment()

    utilities = Utilities(logger=logger)

    trained_model = utilities.load_model()

    data = utilities.load_data()

    sampler = Sampler()

    if not trained_model:
        if data is not None and not data.empty:
            trained_model = sampler.train_model(data)
            utilities.save_model(trained_model)

        else:
            data = sampler.generate_dataset()
            utilities.save_data(data)

            trained_model = sampler.train_model(data)
            utilities.save_model(trained_model)

    env_train, env_test = env.initialize_env(model=trained_model)

    agent = Agent(
        env_train=env_train,
        env_test=env_test
    )

    agent.run_agent(
        env_train=env_train,
        env_test=env_test,
        discount_factor=args.discount_factor,
        epsilon=args.epsilon,
        entropy_coefficient=args.entropy_coefficient,
        hidden_dimensions=args.hidden_dimensions,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        reward_threshold=args.reward_threshold,
        ppo_steps=args.ppo_steps,
        plot=True
    )

    torch.save({
        'actor_state_dict': agent.trained_actor.state_dict(),
        'critic_state_dict': agent.trained_critic.state_dict()
    }, model_path)

    print("Actor and Critic models have been saved.")

    """
    optimizer = Optimizer(env=env, agent=agent, model=trained_model)
    optimizer.optimize_hyperparameters()"""
