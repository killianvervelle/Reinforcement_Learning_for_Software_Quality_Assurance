import warnings
from agent import Agent
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

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

warnings.filterwarnings("ignore", category=DeprecationWarning)

np.bool8 = np.bool_

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ["JVM_ARGS"] = "-Dlog_level.jmeter=OFF"


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

    def load_model(self, path):
        try:
            model = torch.load(path)
            print(f"Model loaded successfully from {path}.")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {path}: {e}")

    def initialize_env(self):
        requirement_res_times = [2500, 2500]
        vms = self.initialize_vms(2, requirement_res_times)
        print(f"Initialized {len(vms)} VMs.")

        if "Env-v1" not in gym.envs.registry:
            register(
                id="Env-v1",
                entry_point="my_gym.myGymStress:ResourceStarving"
            )

        env_train = gym.make("Env-v1", vm=vms[0])
        env_test = gym.make("Env-v1", vm=vms[0])

        return env_train, env_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--discount_factor", type=float, default=0.90)
    parser.add_argument("--epsilon", type=float, default=0.31)
    parser.add_argument("--entropy_coefficient", type=float, default=0.08)
    parser.add_argument("--hidden_dimensions", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.22)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument("--model-dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "/app/model"))
    args = parser.parse_args()

    env = Environment()

    utilities = Utilities(logger=logger)

    env_train, env_test = env.initialize_env()

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
        plot=False
    )

    model_path = os.path.join(args.model_dir, "ppo_trained_model.pth")
    torch.save(agent, model_path)
    print(f"Model saved to {model_path}")

    # optimizer = Optimizer(env=env, agent=agent, model=model)
    # optimizer.optimize_hyperparameters()
