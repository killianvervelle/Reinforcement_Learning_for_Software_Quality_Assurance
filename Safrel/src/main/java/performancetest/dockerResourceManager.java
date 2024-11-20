package performancetest;

import com.github.dockerjava.api.DockerClient;
import com.github.dockerjava.core.DefaultDockerClientConfig;
import com.github.dockerjava.core.DockerClientBuilder;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class dockerResourceManager {

    private String containerID; // The name of the container to update
    private final DockerClient dockerClient;

    public dockerResourceManager() throws IOException, InterruptedException {
        DefaultDockerClientConfig config = DefaultDockerClientConfig
                .createDefaultConfigBuilder()
                .withDockerHost("tcp://host.docker.internal:2375")
                .build();

        // Build the Docker Client
        this.dockerClient = DockerClientBuilder.getInstance(config)
                .build();
       this.containerID = getContainerId();
       System.out.println("Container id: " + containerID);
       Thread.sleep(5000);

    }

    private String getContainerId() throws IOException {
        String cgroupFile = "/proc/self/cgroup";
        try (BufferedReader reader = new BufferedReader(new FileReader(cgroupFile))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.contains("/docker/")) {
                    return line.substring(line.lastIndexOf("/") + 1);
                }
            }
        }
        throw new IOException("Container ID not found in " + cgroupFile);
    }

    public void updateContainerCpuUtil(int cpuQuota) {
        try {
            dockerClient.updateContainerCmd(containerID)
                    .withCpuQuota(cpuQuota*1000)
                    .exec();
            System.out.println("Updated CPU utilization to " + cpuQuota + "%");
        } catch (Exception e) {
            System.err.println("Failed to update CPU utilization: " + e.getMessage());
        }
    }

    public void updateContainerMemoryUtil(long memoryInGB, long swapMemoryInGB) {
        try {
            long memoryInBytes = memoryInGB * 1024L * 1024L * 1024L; // Convert GB to bytes
            dockerClient.updateContainerCmd(containerID)
                    .withMemory(memoryInBytes)
                    .withMemorySwap(swapMemoryInGB)
                    .exec();
            System.out.println("Updated memory utilization to " + memoryInGB + " GB");
        } catch (Exception e) {
            System.err.println("Failed to update memory utilization: " + e.getMessage());
        }
    }

    public void updateContainerDiskUtil(long diskQuotaInGB) {
        try {
            dockerClient.updateContainerCmd(containerID)
                    .withMemory(diskQuotaInGB * 1024L * 1024L * 1024L)
                    .exec();
            System.out.println("Updated disk quota to " + diskQuotaInGB + " GB");
        } catch (Exception e) {
            System.err.println("Failed to update disk quota: " + e.getMessage());
        }
    }

}
