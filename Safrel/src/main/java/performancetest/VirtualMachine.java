package performancetest;


public class VirtualMachine {
    public double Throughput;
    public double ResponseTime_i;
    public double ResponseTime;
    public double NormalizedResponsetime;
    public double Requirement_ResTime;
    public double Acceptolerance;
    // 4 cores with 2.5 GHz
    public int VM_CPU_g;
    public long VM_Mem_g;
    public long VM_Disk_g;
    public int VM_CPU_i;
    public long VM_Mem_i;
    public long VM_Disk_i;
    public double VM_CPUtil;
    public double VM_MemUtil;
    public double VM_DiskUtil;
    public Double[] VM_SensitivityValues;


    public void CalculateCPUtilImprov() {
        this.VM_CPUtil= (double) this.VM_CPU_g /this.VM_CPU_i;
    }
    
    public void CalculateMemUtilImprov() {
        this.VM_MemUtil= (double) this.VM_Mem_g /this.VM_Mem_i;
    }
    
    public void CalculateDiskUtilImprov() {
        this.VM_DiskUtil= (double) this.VM_Disk_g /this.VM_Disk_i;
    }
    
    public void NormalizeResponseTime() {
        // R_Norm= b.2/Math.Pi
        this.NormalizedResponsetime= (2.0/Math.PI) * Math.atan(((this.ResponseTime/this.Requirement_ResTime)));
    }
}
