import random
import time
import numpy as np

import torch.optim as optim

from model.VirtualMachine import VirtualMachine
from model.A2C import A2C

from my_gym.envs.StressEnv import StressingEnvironment


class StressTesting:
    """
    Class for performing stress testing by initializing a set of Virtual Machines (VMs).
    """

    def __init__(self):
        self.learning_rate = 1
        self.maximum_episodes = 100

    def main(self):
        """
        Main method for stress testing. 

        This method configures and initializes a set of Virtual Machines (VMs) with various 
        resource limits, response time requirements, and sensitivity values. It serves as the 
        entry point for setting up the environment for stress testing.

        Steps:
            1. Defines resource limits and sensitivity profiles for the VMs.
            2. Initializes a predefined number of VMs with randomized capacities and configurations.
            3. Prints the number of initialized VMs for verification.

        Example:
            Running this method will initialize VMs and begin the reinforcement learning process.
        """
        requirement_res_times = [3000, 3500, 4000, 4500, 5000, 5000]
        sensitivity_collection = [
            [0.79, 0.19, 0.01],
            [0.74, 0.24, 0.01],
            [0.01, 0.74, 0.25],
            [0.27, 0.53, 0.20],
            [0.53, 0.46, 0.01],
            [0.32, 0.01, 0.67],
        ]

        vms = self.initialize_vms(
            10, 100, 100, 100, requirement_res_times, sensitivity_collection)
        print(f"Initialized {len(vms)} VMs.")

        # Running the stress test on the first VM.
        vm = vms[0]
        agent = A2C(hidden_size=64,
                    gamma=0.99,
                    vm=vm,
                    random_seed=None)

        actor_optim = optim.Adam(
            agent.actor.parameters(), lr=self.learning_rate)
        critic_optim = optim.Adam(
            agent.critic.parameters(), lr=self.learning_rate)

        r = []
        avg_r = []
        l_actor = []
        l_critic = []

        for i in range(self.maximum_episodes):

            critic_optim.zero_grad()
            actor_optim.zero_grad()

            rewards, critic_vals, actor_probs, total_reward = agent.train_model(
                render=False)
            r.append(total_reward)

            normalized_rewards = self.normalize(rewards)
            normalized_critic_vals = self.normalize(critic_vals)

            loss_actor, loss_critic = agent.compute_loss(
                actor_probs=actor_probs, actual_returns=normalized_rewards, expected_returns=normalized_critic_vals)

            print("LOSS ACTOR, LOSS CRITIC", loss_actor, loss_critic)
            actor_optim.zero_grad()
            loss_actor.backward()

            critic_optim.zero_grad()
            loss_critic.backward()

            l_actor.append(loss_actor)
            l_critic.append(loss_critic)

            # smoothing randomness in reward fluctuations
            if i % 50 == 0 and i > 0:
                reward_subset = r[i-50:i]  # Get the last 50 rewards
                average_reward = sum(reward_subset) / len(reward_subset)
                avg_r.append(average_reward)
                print(
                    f"Average reward during episodes {i-50}-{i} is {average_reward}.")

        for _ in range(20):
            time.sleep(0.5)
            agent.test_model(render=True)

    @staticmethod
    def normalize(tensor, epsilon=1e-8):
        """ Normalize the tensor to have zero mean and unit variance """
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / (std + epsilon)

    @staticmethod
    def initialize_vms(n, vms_cap_cpu, vms_cap_mem, vms_cap_disk, requirement_res_times, sensitivity_collection):
        """
        Initializes a set of Virtual Machines (VMs) for stress testing.

        This function creates `n` Virtual Machine objects, each with randomized capacities for CPU, memory, 
        and disk usage, as well as specific sensitivity values and response time requirements. These VMs 
        are configured with both initial and current resource utilization and response time values.

        Args:
            n (int): The number of Virtual Machines to initialize.
            vms_cap_cpu (float): The upper capacity limit for VM CPU.
            vms_cap_mem (float): The upper capacity limit for VM memory.
            vms_cap_disk (float): The upper capacity limit for VM disk.
            requirement_res_times (list of float): A list of possible response time requirements.
            sensitivity_collection (list of list of float): A collection of sensitivity values 
                                                            for CPU, memory, and disk for each VM.
            vm_list (list of VirtualMachine): The list where initialized VirtualMachine objects are added.

        Returns:
            None: The function populates the `vm_list` provided as an argument with initialized VirtualMachine objects.
        """
        vm_list = []

        for i in range(n):
            vm_cpu = random.randint(60, int(vms_cap_cpu))
            vm_mem = random.randint(60, int(vms_cap_mem))
            vm_disk = random.randint(60, int(vms_cap_disk))
            vm_sensitivity_index = random.randint(
                0, len(sensitivity_collection) - 1)
            vm_required_res_time_index = random.randint(
                0, len(requirement_res_times) - 1)

            vm_res_time = random.uniform(
                requirement_res_times[vm_required_res_time_index] / 1.2,
                requirement_res_times[vm_required_res_time_index] / 1.5
            )

            print(f"VM{i}")
            print(f"Initial Response Time: {vm_res_time}")
            print(
                f"Required Response Time: {requirement_res_times[vm_required_res_time_index]}")

            vm = VirtualMachine(
                cpu_g=vm_cpu,
                mem_g=vm_mem,
                disk_g=vm_disk,
                cpu_i=vm_cpu,
                mem_i=vm_mem,
                disk_i=vm_disk,
                sensitivity_values=sensitivity_collection[vm_sensitivity_index],
                response_time_i=vm_res_time,
                Requirement_ResTime=requirement_res_times[vm_required_res_time_index],
                Acceptolerance=0.1
            )

            vm_list.append(vm)

        return vm_list


if __name__ == "__main__":

    stress_tester = StressTesting()
    stress_tester.main()
