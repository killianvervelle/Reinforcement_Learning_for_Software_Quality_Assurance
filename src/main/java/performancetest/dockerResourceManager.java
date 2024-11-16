package performancetest;

import com.github.dockerjava.api.DockerClient;
import com.github.dockerjava.core.DockerClientBuilder;


public class dockerResourceManager {

    private final String containerName; // The name of the container to update
    private final DockerClient dockerClient;

    public dockerResourceManager(String containerName) {
        this.containerName = containerName;
        this.dockerClient = DockerClientBuilder.getInstance().build();
    }

    public void updateContainerCpuUtil(int cpuQuota) {
        try {
            dockerClient.updateContainerCmd(this.containerName)
                    .withCpuQuota(cpuQuota*1000)
                    .exec();
            System.out.println("Updated CPU utilization to " + cpuQuota + "%");
        } catch (Exception e) {
            System.err.println("Failed to update CPU utilization: " + e.getMessage());
        }
    }

    public void updateContainerMemoryUtil(long memoryInGB) {
        try {
            long memoryInBytes = memoryInGB * 1024L * 1024L * 1024L; // Convert GB to bytes
            dockerClient.updateContainerCmd(this.containerName)
                    .withMemory(memoryInBytes)
                    .exec();
            System.out.println("Updated memory utilization to " + memoryInGB + " GB");
        } catch (Exception e) {
            System.err.println("Failed to update memory utilization: " + e.getMessage());
        }
    }

    public void updateContainerDiskUtil(long diskQuotaInGB) {
        try {
            dockerClient.updateContainerCmd(this.containerName)
                    .withMemory(diskQuotaInGB * 1024L * 1024L * 1024L)
                    .exec();
            System.out.println("Updated disk quota to " + diskQuotaInGB + " GB");
        } catch (Exception e) {
            System.err.println("Failed to update disk quota: " + e.getMessage());
        }
    }

}
