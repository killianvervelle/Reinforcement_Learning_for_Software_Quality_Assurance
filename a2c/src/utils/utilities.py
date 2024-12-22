import numpy as np

import torch
from torch import nn

from model.virtualMachine import VirtualMachine


class Utilities:
    def __init__(self):
        pass

    def discounted_rewards(self, rewards, dones, gamma):
        """
        Purpose: Computes the discounted cumulative rewards for a sequence of rewards, 
        taking into account terminal states indicated by `dones`.
        """
        ret = 0
        discounted = []
        for reward, done in zip(rewards[::-1], dones[::-1]):
            ret = reward + ret * gamma * (1-done)
            discounted.append(ret)

        return discounted[::-1]

    def process_memory(self, memory, gamma=0.99, discount_rewards=True):
        """
        Purpose: Processes a list of experience tuples from an agent's memory into separate 
        tensors for actions, states, next states, rewards, and dones. Optionally applies 
        discounting to rewards.
        """
        actions = []
        states = []
        next_states = []
        rewards = []
        dones = []

        for action, reward, state, next_state, done in memory:
            actions.append(action)
            rewards.append(reward)
            states.append(state)
            next_states.append(next_state)
            dones.append(done)

        if discount_rewards:
            if False and dones[-1] == 0:
                rewards = discounted_rewards(
                    rewards + [last_value], dones + [0], gamma)[:-1]
            else:
                rewards = self.discounted_rewards(rewards, dones, gamma)

        actions = self.to_tensor(actions).view(-1, 1)
        states = self.to_tensor(states)
        next_states = self.to_tensor(next_states)
        rewards = self.to_tensor(rewards).view(-1, 1)
        dones = self.to_tensor(dones).view(-1, 1)
        return actions, rewards, states, next_states, dones

    def clip_grad_norm_(self, module, max_grad_norm):
        """
        Purpose: Clips the gradients of the parameters in the optimizer to prevent exploding gradients 
        during backpropagation.
        """
        nn.utils.clip_grad_norm_(
            [p for g in module.param_groups for p in g["params"]], max_grad_norm)

    def to_tensor(self, x):
        """
        Purpose:  Purpose: Processes the input `x` based on its type, converting it into a tensor format that 
        can be used by a neural network.
        """
        if isinstance(x, list) and isinstance(x[0], dict):
            x = np.array([[item['agent'], item['target']]
                         for item in x], dtype=np.float32)
        if isinstance(x, dict):
            x = np.array(list(x.values()), dtype=np.float32)
        if isinstance(x, tuple):
            array_part, dict_part = x
            if not isinstance(array_part, np.ndarray):
                raise ValueError(
                    f"Expected first element of tuple to be ndarray, got {type(array_part)}")
            x = array_part
        elif not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.float32)
        return torch.from_numpy(x).float()

    def init_xavier_weights(self, m):
        """
        Purpose: Initializes the weights of a given layer using the Xavier (Glorot) initialization method. 
        This method is used for layers in deep neural networks to help with training by maintaining the 
        variance of activations across layers.
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def normalize_rewards(self, r):
        mean = np.mean(r)
        std = np.std(r) + 1e-8
        return (r - mean) / std

    def initialize_vms(self, n, vms_cap_cpu, vms_cap_mem, vms_cap_disk, requirement_res_times, sensitivity_collection):
        """
        Purpose:
            Initialize a list of virtual machine (VM) instances with predefined configurations and return the list.

        Arguments:
            n (int): The number of virtual machines to initialize.
            vms_cap_cpu (float): The CPU capacity for the VMs.
            vms_cap_mem (float): The memory capacity for the VMs.
            vms_cap_disk (float): The disk capacity for the VMs.
            requirement_res_times (list): A list of required response times for the VMs.
            sensitivity_collection (list): A collection of sensitivity values to initialize the VMs.

        Returns:
            list: A list containing `n` initialized `VirtualMachine` instances.
        """
        vm_list = []

        for i in range(n):
            vm_cpu = 18
            vm_mem = 10
            vm_disk = 40
            vm_sensitivity_index = 0
            vm_required_res_time_index = 3300

            vm_res_time = 2900

            print(f"VM{i}")
            print(f"Initial Response Time: {vm_res_time}")
            print(
                f"Required Response Time: {3300}")

            vm = VirtualMachine(
                cpu_g=vm_cpu,
                mem_g=vm_mem,
                disk_g=vm_disk,
                cpu_i=vm_cpu,
                mem_i=vm_mem,
                disk_i=vm_disk,
                sensitivity_values=sensitivity_collection[vm_sensitivity_index],
                response_time_i=3050,
                Requirement_ResTime=3300,
                Acceptolerance=0.1
            )

            vm_list.append(vm)

        return vm_list
