import numpy as np
import torch
import gym
import logging
from gym.envs import register

from agent import Agent
from virtualMachine import VirtualMachine
from optimizer import Optimizer
from a2c.src.utilities import Utilities

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# bool8 still required by gym for some reason
np.bool8 = np.bool_

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Environment:
    def __init__(self):
        pass

    def initialize_vms(self, n, requirement_res_times, model):
        vm_list = []

        for i in range(n):
            vm = VirtualMachine(
                Requirement_ResTime=requirement_res_times[i],
                model=model
            )
            vm_list.append(vm)

        return vm_list

    def load_model(self, path):
        try:
            model = torch.load(path)
            print(f"Model loaded successfully from {path}.")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {path}: {e}")

    def initialize_env(self, model):
        requirement_res_times = [2500, 2500]
        vms = self.initialize_vms(2, requirement_res_times, model)
        print(f"Initialized {len(vms)} VMs.")

        if "Env-v1" not in gym.envs.registry:
            register(
                id="Env-v1",
                entry_point="my_gym.myGym:ResourceStarving"
            )

        env_train = gym.make("Env-v1", vm=vms[0])
        env_test = gym.make("Env-v1", vm=vms[0])

        return env_train, env_test


if __name__ == "__main__":
    env = Environment()

    utilities = Utilities(logger=logger)

    model = utilities.load_model()

    env_train, env_test = env.initialize_env(model)

    agent = Agent(
        env_train=env_train,
        env_test=env_test
    )

    agent.run_agent(
        env_train=env_train,
        env_test=env_test,
        discount_factor=0.90,
        entropy_coefficient=0.08,
        hidden_dimensions=64,
        dropout=0.22,
        batch_size=32,
        learning_rate=0.0003,
        max_grad_norm=10.0,
        plot=True
    )

    # optimizer = Optimizer(env=env, agent=agent, model=model)
    # optimizer.optimize_hyperparameters()
