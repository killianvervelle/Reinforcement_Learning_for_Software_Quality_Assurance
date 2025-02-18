import numpy as np
import requests
import os

import gym
import gym.spaces as Space


class ResourceStarving(gym.Env):
    """
    Purpose: 
        This class represents a custom reinforcement learning environment for simulating the management of virtual machine (VM) 
        resources (CPU, memory, disk) and ensuring that the VM's response time meets the target requirement. The agent can adjust 
        the resources allocated to the VM and receive feedback on performance.

    Arguments:
        vm (VMClass): An instance of a virtual machine class containing the initial configuration of the VM (e.g., CPU, memory, disk, response time).
        render_mode (str, optional): The rendering mode to use. If provided, it should be either "human" or "rgb_array". Default is None.

    Output:
        None: Initializes the environment's parameters.
    """

    def __init__(self, vm, model, render_mode=None) -> None:
        super(ResourceStarving, self).__init__()

        self.window_size = 500
        self.vm = vm
        self.model = model

        high = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        low = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

        self.action_space = gym.spaces.Discrete(4)

        self.action_to_adjustment = [
            (-1, 0),
            (1, 0),
            (0, -0.05),
            (0, 0.05),
        ]

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def step(self, action):
        """
        Executes an action in the environment, updates the system state, and calculates the reward.
        """
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        cpu_adjustment, mem_adjustment = self.action_to_adjustment[action]
        assert any([cpu_adjustment, mem_adjustment]
                   ), "Invalid action: all adjustments are set to zero."

        # Adjusting the system's resources
        if cpu_adjustment != 0:
            self.vm.VM_CPU_g = self.vm.VM_CPU_g + cpu_adjustment
        if mem_adjustment != 0:
            self.vm.VM_Mem_g = self.vm.VM_Mem_g + mem_adjustment

        # Computing the system's resonse time by executing Jmeter's test plan
        self.vm.ResponseTime = int(self.model.predict(
            [[self.vm.VM_CPU_g, self.vm.VM_Mem_g]])[0])

        # Computing the system's resource utilization ratios
        cpu_util = self.vm.VM_CPU_g / self.vm.VM_CPU_i
        mem_util = self.vm.VM_Mem_g / self.vm.VM_Mem_i
        response_time_norm = self.vm.ResponseTime / self.vm.Requirement_ResTime

        # Setting the new system state
        self.state = (
            round(cpu_util, 2),
            round(mem_util, 2),
            round(response_time_norm, 2)
        )

        terminated = bool(
            response_time_norm > 1.0 or
            cpu_util > 1.1 or cpu_util < 0.4 or
            mem_util > 1.1 or mem_util < 0.4
        )

        # Setting the termination conditions of an episode
        if terminated:
            reward = -1.0  # Penalize termination
        else:
            # Reward for maintaining response time close to the target
            reward = 1.0 - abs(response_time_norm - 1.0)

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        self.vm.reset()
        self.vm.ResponseTime = int(self.model.predict(
            [[self.vm.VM_CPU_g, self.vm.VM_Mem_g]])[0])

        cpu_util = self.vm.VM_CPU_g / self.vm.VM_CPU_i
        mem_util = self.vm.VM_Mem_g / self.vm.VM_Mem_i
        response_time_norm = self.vm.ResponseTime / self.vm.Requirement_ResTime

        self.state = np.array(
            [cpu_util, mem_util, response_time_norm], dtype=np.float32)
        return self.state, {}
