package performancetest;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;


public class nQueenProblem {
    private static final int N = 29; // Size of the chessboard and number of queens
    private static final int TEST_DURATION_SECONDS = 30; // Test duration in seconds

    private final List<Long> responseTimes = new ArrayList<>();
    private final AtomicInteger successfulRuns = new AtomicInteger(0);
    private final Long averageResponseTime = null;

    public static void main(String[] args) {
        nQueenProblem benchmark = new nQueenProblem();
        benchmark.runBenchmark();
    }

    public void runBenchmark() {
        System.out.println("Starting n-Queens problem benchmark...");
        Instant startTime = Instant.now();

        while (Duration.between(startTime, Instant.now()).getSeconds() < TEST_DURATION_SECONDS) {
            Instant taskStart = Instant.now();

            if (solveNQ(N)) {
                long responseTime = Duration.between(taskStart, Instant.now()).toMillis();
                responseTimes.add(responseTime);
                successfulRuns.incrementAndGet();
                System.out.println("Task completed in: " + responseTime + " ms");
            } else {
                System.out.println("No solution found.");
            }
        }

        calculateAndPrintResults();
    }

    public boolean solveNQ(int size) {
        int[][] board = new int[size][size];
        return solveNQUtil(board, 0, size);
    }

    private boolean solveNQUtil(int[][] board, int col, int size) {
        if (col >= size) return true;

        for (int i = 0; i < size; i++) {
            if (isSafe(board, i, col, size)) {
                board[i][col] = 1;

                if (solveNQUtil(board, col + 1, size)) {
                    return true;
                }

                board[i][col] = 0; // BACKTRACK
            }
        }

        return false;
    }

    private boolean isSafe(int[][] board, int row, int col, int size) {
        for (int i = 0; i < col; i++) if (board[row][i] == 1) return false;

        for (int i = row, j = col; i >= 0 && j >= 0; i--, j--) if (board[i][j] == 1) return false;

        for (int i = row, j = col; i < size && j >= 0; i++, j--) if (board[i][j] == 1) return false;

        return true;
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
