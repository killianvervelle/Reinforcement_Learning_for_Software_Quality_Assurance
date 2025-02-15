import pygame
import random
import numpy as np

import gym
import gym.spaces as Space

from asset.gauge import Gauge


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

    def __init__(self, vm, render_mode=None) -> None:
        super(ResourceStarving, self).__init__()

        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": 20
        }
        self.window_size = 500
        self.vm = vm

        high = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        low = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

        # Action Space: 4 discrete actions
        # 0: Decrease CPU, 1: Increase CPU, 2: Decrease Memory, 3: Increase Memory
        self.action_space = gym.spaces.Discrete(4)

        # Map actions to adjustments
        self.action_to_adjustment = [
            (-1, 0),  # Decrease CPU
            (1, 0),   # Increase CPU
            (0, -0.1),  # Decrease Memory
            (0, 0.1)    # Increase Memory
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
            self.vm.VM_CPU_g = max(1, self.vm.VM_CPU_g + cpu_adjustment)
        if mem_adjustment != 0:
            self.vm.VM_Mem_g = max(0.1, self.vm.VM_Mem_g + mem_adjustment)

        # Computing the system's theoritical response time after the adjustement has been applied
        self.vm.predict_responsetime()

        # Computing the system's resource utilization ratios
        cpu_util = self.vm.VM_CPU_g / self.vm.VM_CPU_i
        mem_util = self.vm.VM_Mem_g / self.vm.VM_Mem_i
        response_time_norm = self.vm.ResponseTime / self.vm.Requirement_ResTime

        # Setting the new system state
        self.state = (
            cpu_util,
            mem_util,
            response_time_norm
        )

        terminated = bool(
            response_time_norm > 1.1 or
            cpu_util > 1.0 or cpu_util < 0.1 or
            mem_util > 1.0 or mem_util < 0.1
        )

        # Setting the termination conditions of an episode
        if terminated:
            reward = -1.0  # Penalize termination
        else:
            # Reward for maintaining response time close to the target
            reward = 1.0 - abs(response_time_norm - 1.0)

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def render(self):
        if self.render_mode == "rgb_array":
            return self.render_frame()

    def render_frame(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        if self.render_mode == "rgb_array":
            self.window.fill((255, 255, 255))

            FONT = pygame.font.SysFont('Arial', 30)
            my_gauge = Gauge(
                screen=self.window,
                FONT=FONT,
                x_cord=self.window_size / 2,
                y_cord=self.window_size / 2,
                thickness=30,
                radius=150,
                circle_colour=(128, 128, 128),
                glow=False
            )

            percent = (self.vm.ResponseTime /
                       int(1.1 * self.vm.Requirement_ResTime)) * 100
            my_gauge.draw(percent)

            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def reset(self, seed=None, options=None):
        self.vm.reset()
        cpu_util = self.vm.VM_CPU_g / self.vm.VM_CPU_i
        mem_util = self.vm.VM_Mem_g / self.vm.VM_Mem_i
        response_time_norm = self.vm.ResponseTime / self.vm.Requirement_ResTime

        self.state = np.array(
            [cpu_util, mem_util, response_time_norm], dtype=np.float32)
        return self.state, {}

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
