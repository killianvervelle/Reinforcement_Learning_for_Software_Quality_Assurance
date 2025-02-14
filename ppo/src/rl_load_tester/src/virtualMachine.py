import random
from dotenv import load_dotenv
import os


class VirtualMachine:
    """
    Purpose:
        This class represents a virtual machine (VM) with specified hardware resources and sensitivity values.
        It provides functionality to calculate throughput and response time based on CPU, memory, and disk resources.

    """

    def __init__(self, Requirement_ResTime):
        self.VM_CPU_i = 0
        self.VM_Mem_i = 0.0
        self.VM_CPU_g = self.VM_CPU_i
        self.VM_Mem_g = self.VM_Mem_i
        self.ResponseTime_i = 0
        self.Requirement_ResTime = Requirement_ResTime
        self.ResponseTime = 0

        self.threads = int(os.getenv("THREADS", 5))
        self.rampup = int(os.getenv("RAMPUP", 1))
        self.loops = int(os.getenv("LOOPS", 5))

    def reset(self):
        self.VM_CPU_i = 100
        self.VM_Mem_i = 1.0
        self.VM_CPU_g = self.VM_CPU_i
        self.VM_Mem_g = self.VM_Mem_i
        self.ResponseTime_i = 0
        self.ResponseTime = self.ResponseTime_i

        self.threads = int(os.getenv("THREADS", 3))
        self.rampup = int(os.getenv("RAMPUP", 5))
        self.loops = int(os.getenv("LOOPS", 5))
