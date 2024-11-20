package performancetest;

import java.time.Duration;
import java.time.Instant;
import java.util.Random;


public class dummyMemTask implements benchmark {

    private static final int ARRAY_SIZE = 50_000_000;
    private static final Random RANDOM = new Random();

    @Override
    public double runBenchmark(VirtualMachine VM) {
        System.out.println("Starting memory-heavy task simulation...");
        Instant taskStart = Instant.now();

        try {
            performMemoryHeavyOperation();
            long responseTime = Duration.between(taskStart, Instant.now()).toMillis();
            System.out.println("Task completed in: " + responseTime + " ms");
            return responseTime;
        } catch (OutOfMemoryError e) {
            System.err.println("Memory overflow during task execution: " + e.getMessage());
            return -1;
        }
    }

    private void performMemoryHeavyOperation() {
        double[] largeArray = new double[ARRAY_SIZE];
        double sum = 0;

        for (int i = 0; i < ARRAY_SIZE; i++) {
            largeArray[i] = RANDOM.nextDouble();
            sum += largeArray[i];
        }
        Math.sqrt(sum);
    }

}
