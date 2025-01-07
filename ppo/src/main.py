import numpy as np

import gym
from gym.envs import register

from model.agent import Agent
from model.virtualMachine import VirtualMachine

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# bool8 still required by gym for some reason
np.bool8 = np.bool_


class Simulate:
    """
    Purpose:
        This class manages the simulation of virtual machines (VMs) and reinforcement learning environments. 
        It initializes VMs, registers the custom environment, and runs the simulation loop involving an agent.

    Methods:
        _register_env: Registers the custom environment 'Env-v0' if it is not already registered in Gym.
        initialize_vms: Initializes a set of virtual machines with specific characteristics.
        initialize_environments: Creates training and testing environments for reinforcement learning.
        run_simulation: Executes the simulation process by initializing VMs, environments, and running the agent.
    """

    def __init__(self):
        self._register_env()

    def _register_env(self):
        if "Env-v0" not in gym.envs.registry:
            register(
                id="Env-v0",
                # Ensure Environment class is properly defined elsewhere
                entry_point="my_gym.myGym:Environment",
                kwargs={"render_mode": "rgb_array"}
            )

    def initialize_vms(self, n, vms_cap_cpu, vms_cap_mem, vms_cap_disk):
        requirement_res_times = [3000, 3050, 3100, 3100]
        sensitivity_collection = [
            [0.79, 0.19, 0.01],
            [0.74, 0.24, 0.01],
            [0.27, 0.63, 0.10],
            [0.42, 0.50, 0.07]
        ]
        vm_list = []

        for i in range(n):
            vm_cpu = vms_cap_cpu
            vm_mem = vms_cap_mem
            vm_disk = vms_cap_disk
            vm_res_time = 0.0
            vm_sensitivities = sensitivity_collection[i]
            vm_required_time = requirement_res_times[i]

            vm = VirtualMachine(
                cpu_g=vm_cpu,
                mem_g=vm_mem,
                disk_g=vm_disk,
                cpu_i=vm_cpu,
                mem_i=vm_mem,
                disk_i=vm_disk,
                sensitivity_values=vm_sensitivities,
                response_time_i=vm_res_time,
                Requirement_ResTime=vm_required_time,
                Acceptolerance=0.1
            )

            vm_list.append(vm)
        return vm_list

    def initialize_environments(self, vm):
        env_train = gym.make("Env-v0", vm=vm, render_mode="rgb_array")
        env_test = gym.make("Env-v0", vm=vm, render_mode="rgb_array")
        return env_train, env_test

    def run_simulation(self):
        vms = self.initialize_vms(4, 25, 25, 100)
        print(f"Initialized {len(vms)} VMs.")
        vm = vms[0]

        env_train, env_test = self.initialize_environments(vm)
        agent = Agent(env_train, env_test)
        agent.run_agent()


if __name__ == "__main__":
    simulation = Simulate()
    simulation.run_simulation()
