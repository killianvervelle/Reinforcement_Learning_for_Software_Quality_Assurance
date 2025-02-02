import random


class VirtualMachine:
    """
    Purpose:
        This class represents a virtual machine (VM) with specified hardware resources and sensitivity values.
        It provides functionality to calculate throughput and response time based on CPU, memory, and disk resources.

    """

    def __init__(self, Requirement_ResTime, model):
        self.VM_CPU_i = 0
        self.VM_Mem_i = 0.0
        self.VM_CPU_g = self.VM_CPU_i
        self.VM_Mem_g = self.VM_Mem_i
        self.ResponseTime_i = 0
        self.Requirement_ResTime = Requirement_ResTime
        self.ResponseTime = 0
        self.model = model

    def predict_responsetime(self):
        self.ResponseTime = int(self.model.predict(
            [[self.VM_CPU_g, self.VM_Mem_g]])[0])
        return self.ResponseTime

    def reset(self):
        self.VM_CPU_i = random.randint(90, 100)
        self.VM_Mem_i = random.uniform(1.0, 1.2)
        self.VM_CPU_g = self.VM_CPU_i * 0.8
        self.VM_Mem_g = self.VM_Mem_i * 0.8
        self.ResponseTime_i = self.predict_responsetime()
        self.ResponseTime = self.ResponseTime_i
