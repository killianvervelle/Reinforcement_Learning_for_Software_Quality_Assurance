import os


class VirtualMachine:
    def __init__(self, Requirement_ResTime):
        self.VM_CPU_i = 0
        self.VM_Mem_i = 0.0
        self.VM_CPU_g = self.VM_CPU_i
        self.VM_Mem_g = self.VM_Mem_i
        self.ResponseTime_i = 0
        self.Requirement_ResTime = Requirement_ResTime
        self.ResponseTime = 0

        self.threads = int(os.getenv("THREADS", 2))
        self.rampup = int(os.getenv("RAMPUP", 1))
        self.loops = int(os.getenv("LOOPS", 1))

    def reset(self):
        self.VM_CPU_i = 90
        self.VM_Mem_i = 1.8
        self.VM_CPU_g = self.VM_CPU_i
        self.VM_Mem_g = self.VM_Mem_i
        self.ResponseTime_i = 0
        self.ResponseTime = 0

        self.threads = int(os.getenv("THREADS", 2))
        self.rampup = int(os.getenv("RAMPUP", 1))
        self.loops = int(os.getenv("LOOPS", 1))
