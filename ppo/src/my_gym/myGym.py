import pygame
import random
import numpy as np

import gym
import gym.spaces as Space

from asset.gauge import Gauge


class Environment(gym.Env):
    """
    Purpose: 
        This class represents a custom reinforcement learning environment for simulating the management of virtual machine (VM) 
        resources (CPU, memory, disk) and ensuring that the VM's response time meets the target requirement. The agent can adjust 
        the resources allocated to the VM and receive feedback on performance.

    Arguments:
        vm (VMClass): An instance of a virtual machine class containing the initial configuration of the VM (e.g., CPU, memory, disk, response time).
        render_mode (str, optional): The rendering mode to use. If provided, it should be either "human" or "rgb_array". Default is None.

    Output:
        None
    """

    def __init__(self, vm, render_mode=None) -> None:
        super(Environment, self).__init__()

        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": 20
        }
        self.window_size = 500
        self.vm = vm

        high = np.array(
            [
                1.0,
                1.0,
                1.0,
                self.vm.Requirement_ResTime
            ],
            dtype=np.float32,
        )

        low = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0
            ],
            dtype=np.float32,
        )

        self.observation_space = Space.Box(low, high, dtype=np.float32)

        self.action_space = Space.Discrete(4)

        # Mapping actions to the adjustement we will make to the allocated resources
        self.action_to_adjustement = [
            (-1, 0, 0),
            (0, -1, 0),
            (-3, 0, 0),
            (0, -3, 0)
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
        cpu_adjustment, mem_adjustment, disk_adjustment = self.action_to_adjustement[action]
        assert any([cpu_adjustment, mem_adjustment, disk_adjustment]
                   ), "Invalid action: all adjustments are set to zero."

        # Adjusting the system's resources
        if cpu_adjustment != 0:
            self.vm.VM_CPU_g = max(1, self.vm.VM_CPU_g + cpu_adjustment)
        if mem_adjustment != 0:
            self.vm.VM_Mem_g = max(1, self.vm.VM_Mem_g + mem_adjustment)
        if disk_adjustment != 0:
            self.vm.VM_Disk_g = max(1, self.vm.VM_Disk_g + disk_adjustment)

        # Computing the system's theoritical response time after the adjustement has been applied
        self.vm.calculate_throughput_response_time()

        # Computing the system's resource utilization ratios
        cpu_util = self.vm.VM_CPU_g / self.vm.VM_CPU_i
        mem_util = self.vm.VM_Mem_g / self.vm.VM_Mem_i
        disk_util = self.vm.VM_Disk_g / self.vm.VM_Disk_i

        # Setting the new system state
        self.state = (
            cpu_util,
            mem_util,
            disk_util,
            self.vm.ResponseTime
        )

        # Setting the termination conditions of an episode
        terminated = np.bool(
            self.vm.ResponseTime >= self.vm.Requirement_ResTime or
            cpu_util > 1.1 or
            cpu_util < 0.1 or
            mem_util > 1.1 or
            mem_util < 0.1 or
            disk_util > 1.1 or
            disk_util < 0.1
        )

        if terminated:
            reward = 0
        elif self.vm.ResponseTime <= self.vm.Requirement_ResTime:
            reward = 1
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
        super().reset(seed=seed)
        self.vm.ResponseTime_i = random.randint(
            int(self.vm.Requirement_ResTime * 0.4),
            int(self.vm.Requirement_ResTime * 0.6))
        self.vm.VM_CPU_i = random.randint(
            self.vm.CpuCap - 10, self.vm.CpuCap - 5)
        self.vm.VM_Mem_i = random.randint(
            self.vm.MemCap - 10, self.vm.MemCap - 5)
        self.vm.VM_Disk_i = random.randint(
            self.vm.DiskCap - 50, self.vm.DiskCap - 20)

        self.vm.VM_CPU_g = self.vm.VM_CPU_i
        self.vm.VM_Mem_g = self.vm.VM_Mem_i
        self.vm.VM_Disk_g = self.vm.VM_Disk_i

        state = [
            self.vm.VM_CPU_g / self.vm.VM_CPU_i,
            self.vm.VM_Mem_g / self.vm.VM_Mem_i,
            self.vm.VM_Disk_g / self.vm.VM_Disk_i,
            self.vm.ResponseTime_i
        ]

        if self.render_mode == "human":
            self.render_frame()

        return np.array(state, dtype=np.float32), {}

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
