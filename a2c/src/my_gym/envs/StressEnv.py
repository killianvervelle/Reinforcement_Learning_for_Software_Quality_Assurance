import random
import pygame
import numpy as np
import gym
import gym.spaces as Space


from ..assets.Gauge import Gauge


class StressingEnvironment(gym.Env):
    """
    <--- Description --->

    This Reinforcement Learning environment is designed for stress testing software systems by 
    dynamically adjusting resource usage (CPU, memory , and disk). The environment is built using 
    OpenAI Gym. The goal of this environment is to allow an RL agent to learn how to manage system 
    resources to avoid performance degradation, prevent failure, and identify the system's breaking 
    point under various stress conditions.

    <--- Observation Space --->

    The observation is a `ndarray` with shape `(1,)` where the elements correspond to the following:

    | Num | Observation                           | Unit       |
    |-----|---------------------------------------|------------|
    | 0   | Actual Response time                  |     -      |
    | 0   | Maximum Response time                 |   3000 ms  |

    <--- Action Space --->

    The action is an ndarray with shape (9,), where each element corresponds to the change in resource 
    utilization for CPU, memory, and disk, respectively. The actions represent:

    | Num | Action                                | Value |    Unit    |
    |-----|---------------------------------------|-------|------------|
    | 0   | Decrease CPU usage	                  |  1    | percentage |
    | 1   | Decrease CPU usage                    |  3    | percentage |
    | 2   | Decrease CPU usage                    |  5    | percentage |
    | 3   | Decrease Mem usage                    |  1    | percentage |
    | 4   | Decrease Mem usage                    |  3    | percentage |
    | 5   | Decrease Mem usage                    |  5    | percentage |
    | 6   | Decrease Dis usage                    |  1    | percentage |
    | 7   | Decrease Dis usage                    |  3    | percentage |
    | 8   | Decrease Dis usage                    |  5    | percentage |

    <--- Transition Dynamics --->

    Given an action, the system updates its state as follows:

    CPU, Memory, and Disk Usage: The system resources change based on the agent’s action. For instance, 
    if the agent chooses to increase CPU usage, the CPU usage increases by a fixed amount (e.g., 5%) and 
    similarly for memory and disk.

    CPU_t+1 = CPU_t - action[0]
    Memory_t+1 = Memory_t - action[2]
    Disk_t+1 = Disk_t - action[4]

    Response Time: The response time increases as system resources are consumed, typically represented by 
    a function that models the stress on the system as resources are exhausted. The response time threshold 
    is set to 3000 ms, with a reward decay between 1500 and 3000ms.

    The theoritical calculation of the response time is done as follows:

    public void CalculateVMThroughput_ResponseTime() {
        double Part1 = (this.VM_CPU_g / this.VM_CPU_i) * this.VM_SensitivityValues[0];
        double Part2 = (this.VM_Mem_g / this.VM_Mem_i) * this.VM_SensitivityValues[1];
        double Part3 = (this.VM_Disk_g / this.VM_Disk_i) * this.VM_SensitivityValues[2];
        double Part4 = this.VM_SensitivityValues[0] + this.VM_SensitivityValues[1] + this.VM_SensitivityValues[2];
        this.Throughput = ((Part1 + Part2 + Part3) / Part4) * 1000.0 / this.ResponseTime_i;
        this.ResponseTime = (double) Math.round((1000.0 / this.Throughput) * 100.0) / 100.0;
    }

    <--- Reward --->

    Penalizing Reward: A positive reward of 1 is given at each timestep if the response time hasn't increase and is still 
    bellow the targeted threshold.However, negative reward of 10 and 30 are given if the response time exceeds the threshold 
    or if the system fails.

    Adaptive Reward: The reward decays as the response time starts increasing beyond 1500 ms towards 
    the critical threshold of 3000 ms. As the system gets closer to failure, the reward is progressively 
    reduced, incentivizing the agent to reduce system load before performance exceeds acceptable limits.

    <--- Starting State --->

    The initial resource utilization values are set to 1.

    <--- Episode End --->

    The episode ends if either of the following happens:
    1. Termination: The system's response time exceeds the maximum value of 3000ms.
    2. Truncation: The length of the episode is 999.
    3. Termination: The resource utilization thresholds reach 0.

    <--- Arguments --->

    ```
    gym.make('system_stressing')
    ```

    <--- Version History --->

    * v0: Initial versions release (1.0.0)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 20
    }

    def __init__(self, vm, render_mode=None) -> None:
        super(StressingEnvironment, self).__init__()

        self.window_size = 500

        self.vm = vm
        self.start_decay = 1500  # ms

        self.observation_space = Space.Dict(
            {
                "agent": Space.Box(0,  self.vm.ResponseTime, shape=(1,), dtype=np.float32),
                "target": Space.Box(0,  self.vm.Requirement_ResTime, shape=(1,), dtype=np.float32)
            }
        )

        self.action_space = Space.Discrete(3)

        # mapping actions to the adjustement we will make to the allocated resources
        self.action_to_adjustement = [
            (-1, 0, 0),  # reverse CPU adjustment
            (0, -1, 0),  # reverse memory adjustment
            (0, 0, -1),  # reverse disk adjustment
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

        Args:
            action (int): The action selected by the agent.
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

        print("Sensitivies", self.vm.VM_SensitivityValues)

        info = self.get_info()

        terminated = np.bool(
            self.vm.ResponseTime >= self.vm.Requirement_ResTime or
            self.vm.VM_CPU_g == 1 or
            self.vm.VM_Mem_g == 1 or
            self.vm.VM_Disk_g == 1
        )

        print("terminated", terminated,
              observation["agent"], observation["target"])

        if terminated:
            reward = -100.0
        else:
            if 0.9 * self.vm.Requirement_ResTime <= self.vm.ResponseTime <= 1.1 * self.vm.Requirement_ResTime:
                reward = 100
            elif self.vm.ResponseTime <= self.vm.Requirement_ResTime:
                distance_penalty = abs(
                    self.vm.ResponseTime - self.vm.Requirement_ResTime) / self.vm.Requirement_ResTime
                reward = 1 - distance_penalty

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
