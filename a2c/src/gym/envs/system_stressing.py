import math
from typing import Optional, Union

import numpy as np

import gym
from gym import logger, spaces


class system_stressing(gym.Env):
    """
    <--- Description --->

    This Reinforcement Learning environment is designed for stress testing software systems by 
    dynamically adjusting resource usage (CPU, memory, and disk). The environment is built using 
    OpenAI Gym. The goal of this environment is to allow an RL agent to learn how to manage system 
    resources to avoid performance degradation, prevent failure, and identify the system's breaking 
    point under various stress conditions.

    <--- Observation Space --->

    The observation is a `ndarray` with shape `(3,)` where the elements correspond to the following:

    | Num | Observation                           | Min  | Max  | Unit       |
    |-----|---------------------------------------|------|------|------------|
    | 0   | CPU usage                             |  0   |  1   | percentage |
    | 1   | Memory usage                          |  0   |  1   | percentage |
    | 2   | Disk usage                            |  0   |  1   | percentage |

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
        "render_modes": ["human", "gauge"],
        "render_fps": 50
    }

    def __init__(self, render_mode: Optional[str] = None) -> None:
        self.cpuUtil = 100 # %
        self.memUtil = 100 # %
        self.diskUtil = 100 # %
        self.updateTime = 0.02 # s

        self.maxResponseTime = 3000 # ms
        self.startDecay = 1500 # ms

        self.action_space = spaces.Discrete(9)
        high = np.array([100, 100, 100])
        low = np.array([0, 0, 0])
        self.observation_space = spaces.Box(low, high, dtype=np.int64)

        self.render_mode = render_mode

        self.screen_width = 500
        self.screen_height = 500
        self.screen = None
        self.clock = None
        self.state = None

        self.maxStepsReached = False

    def step(self, action):
        assert(self.action_space.contains(action))

        
