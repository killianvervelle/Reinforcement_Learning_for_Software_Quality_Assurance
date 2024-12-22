import pygame
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
        None: Initializes the environment's parameters.
    """

    def __init__(self, vm, render_mode=None) -> None:
        super(Environment, self).__init__()

        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": 20
        }

        self.window_size = 500

        self.vm = vm

        self.observation_space = Space.Dict(
            {
                "agent": Space.Box(0,  self.vm.ResponseTime, shape=(1,), dtype=np.float32),
                "target": Space.Box(0,  self.vm.Requirement_ResTime, shape=(1,), dtype=np.float32)
            }
        )

        self.action_space = Space.Discrete(4)

        # mapping actions to the adjustement we will make to the allocated resources
        self.action_to_adjustement = [
            (-1, 0, 0),
            (0, -1, 0),
            (1, 0, 0),
            (0, 1, 0),
        ]
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # `self.clock` used to check if the environment is rendered at the correct framerate in human-mode.
        # `self.window` is a reference to the window we draw to.
        self.window = None
        self.clock = None

    def get_obs(self):
        return {"agent": self.vm.ResponseTime,
                "target": self.vm.Requirement_ResTime
                }

    def get_info(self):
        return {
            "cpu": self.vm.VM_CPU_g,
            "memory": self.vm.VM_Mem_g,
            "disk": self.vm.VM_Disk_g,
            "response_time": self.vm.ResponseTime
        }

    def step(self, action):
        """
        Executes an action in the environment, updates the system state, and calculates the reward.
        """
        cpu_adjustment, mem_adjustment, disk_adjustment = self.action_to_adjustement[action]

        print("adjustement", cpu_adjustment, mem_adjustment, disk_adjustment)

        assert any([cpu_adjustment, mem_adjustment, disk_adjustment]
                   ), "Invalid action: all adjustments are zero."

        if cpu_adjustment != 0:
            self.vm.VM_CPU_g = max(1, self.vm.VM_CPU_g + cpu_adjustment)
        if mem_adjustment != 0:
            self.vm.VM_Mem_g = max(1, self.vm.VM_Mem_g + mem_adjustment)
        if disk_adjustment != 0:
            self.vm.VM_Disk_g = max(1, self.vm.VM_Disk_g + disk_adjustment)

        self.vm.calculate_throughput_response_time()

        print("new VM params", self.vm.VM_CPU_g,
              self.vm.VM_Mem_g, self.vm.VM_Disk_g)

        observation = self.get_obs()

        print("observation from stressEnv", observation)

        info = self.get_info()

        terminated = np.bool(
            self.vm.ResponseTime >= self.vm.Requirement_ResTime or
            self.vm.VM_CPU_g > 30 or
            self.vm.VM_Mem_g > 30 or
            self.vm.VM_CPU_g < 2 or
            self.vm.VM_Mem_g < 2
        )

        print("terminated", terminated,
              observation["agent"], observation["target"])

        if terminated:
            reward = -1
        elif self.vm.ResponseTime <= self.vm.Requirement_ResTime:
            efficiency_bonus = 1.5
            resource_penalty = (self.vm.VM_CPU_g + self.vm.VM_Mem_g) / (
                self.vm.VM_CPU_i + self.vm.VM_Mem_i)
            reward = efficiency_bonus - resource_penalty
        return observation, reward, terminated, False, info

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

        self.vm.ResponseTime = self.vm.ResponseTime_i
        self.vm.VM_CPU_g = self.vm.VM_CPU_i
        self.vm.VM_Mem_g = self.vm.VM_Mem_i
        self.vm.VM_Disk_g = self.vm.VM_Disk_i

        if self.render_mode == "human":
            self.render_frame()

        return self.get_obs()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
