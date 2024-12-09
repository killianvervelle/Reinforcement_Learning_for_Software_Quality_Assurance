class VirtualMachine:

    def __init__(self, cpu_g, mem_g, disk_g, cpu_i, mem_i, disk_i, sensitivity_values, response_time_i, Requirement_ResTime, Acceptolerance):
        self.VM_CPU_g = cpu_g
        self.VM_Mem_g = mem_g
        self.VM_Disk_g = disk_g
        self.VM_CPU_i = cpu_i
        self.VM_Mem_i = mem_i
        self.VM_Disk_i = disk_i
        self.VM_SensitivityValues = sensitivity_values
        self.ResponseTime_i = response_time_i
        self.Requirement_ResTime = Requirement_ResTime
        self.Acceptolerance = Acceptolerance
        self.Throughput = 0.0
        self.ResponseTime = 0.0

    def calculate_throughput_response_time(self):
        part1 = (self.VM_CPU_g / self.VM_CPU_i) * self.VM_SensitivityValues[0]
        part2 = (self.VM_Mem_g / self.VM_Mem_i) * self.VM_SensitivityValues[1]
        part3 = (self.VM_Disk_g / self.VM_Disk_i) * \
            self.VM_SensitivityValues[2]
        part4 = sum(self.VM_SensitivityValues)

        # Calculate throughput
        self.Throughput = ((part1 + part2 + part3) / part4) * \
            1000.0 / self.ResponseTime_i

        # Calculate response time
        self.ResponseTime = round((1000.0 / self.Throughput) * 100.0) / 100.0
