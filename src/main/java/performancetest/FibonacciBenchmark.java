package performancetest;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;


public class FibonacciBenchmark {

    private static final int TEST_DURATION_SECONDS = 30; // Test duration in seconds
    private static final int FIBONACCI_NUMBER = 40; // Fibonacci number to compute (higher numbers are more CPU intensive)

    private final List<Long> responseTimes = new ArrayList<>();
    private final AtomicInteger successfulRuns = new AtomicInteger(0);

    public static void main(String[] args) {
        FibonacciBenchmark benchmark = new FibonacciBenchmark();
        benchmark.runBenchmark();
    }

    public void runBenchmark() {
        System.out.println("Starting Fibonacci Sequence benchmark...");
        Instant startTime = Instant.now();

        while (Duration.between(startTime, Instant.now()).getSeconds() < TEST_DURATION_SECONDS) {
            Instant taskStart = Instant.now();

            long result = computeFibonacci(FIBONACCI_NUMBER); // Compute the Fibonacci number
            long responseTime = Duration.between(taskStart, Instant.now()).toMillis();
            responseTimes.add(responseTime);
            successfulRuns.incrementAndGet();

            System.out.println("Task completed in: " + responseTime + " ms, Fibonacci result: " + result);
        }

        calculateAndPrintResults();
    }

    // A CPU-intensive recursive Fibonacci function
    public long computeFibonacci(int n) {
        if (n <= 1) {
            return n;
        }
        return computeFibonacci(n - 1) + computeFibonacci(n - 2);
    }

    private void calculateAndPrintResults() {
        double averageResponseTime = responseTimes.stream()
                .mapToLong(Long::longValue)
                .average()
                .orElse(0);
        int throughput = successfulRuns.get();

        System.out.println("\n=== Benchmark Results ===");
        System.out.println("Total Successful Runs: " + throughput);
        System.out.println("Average Response Time: " + averageResponseTime + " ms");
        System.out.println("=========================");
    }
}

