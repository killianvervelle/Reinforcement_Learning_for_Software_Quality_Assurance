package performancetest.service;

import performancetest.model.VirtualMachine;
import performancetest.benchmark;

import java.time.Duration;
import java.time.Instant;


public class FibonacciBenchmark implements benchmark {

    private static final int FIBONACCI_NUMBER = 43;

    @Override
    public double runBenchmark(VirtualMachine VM) {
        System.out.println("Starting Fibonacci Sequence performance test...");

        Instant taskStart = Instant.now();

        try {
            long result = computeFibonacci(FIBONACCI_NUMBER);
            long responseTime = Duration.between(taskStart, Instant.now()).toMillis();
            System.out.println("Task completed in: " + responseTime + " ms, Fibonacci result: " + result);
            return responseTime;
        } catch (Exception e) {
            System.err.println("Error during Fibonacci computation: " + e.getMessage());
            return -1;
        }
    }

    public long computeFibonacci(int n) {
        if (n <= 1) {
            return n;
        }
        return computeFibonacci(n - 1) + computeFibonacci(n - 2);
    }

}


