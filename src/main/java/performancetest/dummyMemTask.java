package performancetest;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;


public class dummyMemTask {

    private static final int TEST_DURATION_SECONDS = 30; // Test duration in seconds
    private static final int ARRAY_SIZE = 10_000_000; // Size of arrays to process
    private static final Random RANDOM = new Random();

    private final List<Long> responseTimes = new ArrayList<>();
    private final AtomicInteger successfulTasks = new AtomicInteger(0);

    public static void main(String[] args) {
        dummyMemTask test = new dummyMemTask();
        test.runMemoryIntensiveTasks();
    }

    public void runMemoryIntensiveTasks() {
        System.out.println("Starting memory-heavy task simulation...");
        Instant startTime = Instant.now();

        while (Duration.between(startTime, Instant.now()).getSeconds() < TEST_DURATION_SECONDS) {
            Instant taskStart = Instant.now();
            try {
                performMemoryHeavyOperation(); // Simulate the memory-heavy task
                long responseTime = Duration.between(taskStart, Instant.now()).toMillis();
                responseTimes.add(responseTime);
                successfulTasks.incrementAndGet();
                System.out.println("Task completed in: " + responseTime + " ms");
            } catch (OutOfMemoryError e) {
                System.err.println("Memory overflow during task execution: " + e.getMessage());
                break;
            }
        }

        calculateAndPrintResults();
    }

    private void performMemoryHeavyOperation() {
        // Simulate memory-heavy task by creating a large array and performing operations
        double[] largeArray = new double[ARRAY_SIZE];

        // Fill the array with random values and compute their sum
        double sum = 0;
        for (int i = 0; i < ARRAY_SIZE; i++) {
            largeArray[i] = RANDOM.nextDouble();
            sum += largeArray[i];
        }

        // Perform a mock operation (square root of sum)
        Math.sqrt(sum); // Mock heavy operation
    }

    private void calculateAndPrintResults() {
        double averageResponseTime = responseTimes.stream()
                .mapToLong(Long::longValue)
                .average()
                .orElse(0);
        int throughput = successfulTasks.get();

        System.out.println("\n=== Test Results ===");
        System.out.println("Total Successful Tasks: " + throughput);
        System.out.println("Average Response Time: " + averageResponseTime + " ms");
        System.out.println("====================");
    }
}
