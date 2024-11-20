package performancetest.model;


public class VirtualMachine {
    public double Throughput;
    public double ResponseTime_i;
    public double ResponseTime;
    public double NormalizedResponsetime;
    public double Requirement_ResTime; //in ms
    public double Acceptolerance; //in terms of percentage of requirement response time
    public double VM_CPU_g;
    public double VM_Mem_g;
    public double VM_Disk_g;
    public double VM_CPU_i;
    public double VM_Mem_i;
    public double VM_Disk_i;
    public double VM_CPUtil;
    public double VM_MemUtil;
    public double VM_DiskUtil;
    public Double[] VM_SensitivityValues;

    public void CalculateVMThroughput_ResponseTime() {
        double Part1 = (this.VM_CPU_g / this.VM_CPU_i) * this.VM_SensitivityValues[0];
        double Part2 = (this.VM_Mem_g / this.VM_Mem_i) * this.VM_SensitivityValues[1];
        double Part3 = (this.VM_Disk_g / this.VM_Disk_i) * this.VM_SensitivityValues[2];
        double Part4 = this.VM_SensitivityValues[0] + this.VM_SensitivityValues[1] + this.VM_SensitivityValues[2];
        this.Throughput = ((Part1 + Part2 + Part3) / Part4) * 1000.0 / this.ResponseTime_i;
        this.ResponseTime = (double) Math.round((1000.0 / this.Throughput) * 100.0) / 100.0;
    }

    public void CalculateCPUtilImprov() {
        this.VM_CPUtil = this.VM_CPU_i / this.VM_CPU_g;
    }

    public void CalculateMemUtilImprov() {
        this.VM_MemUtil = this.VM_Mem_i / this.VM_Mem_g;
    }

    public void CalculateDiskUtilImprov() {
        this.VM_DiskUtil = this.VM_Disk_i / this.VM_Disk_g;
    }

    public void NormalizeResponseTime() {
        this.NormalizedResponsetime = (2.0 / Math.PI) * Math.atan(this.ResponseTime / this.Requirement_ResTime);
    }

}
