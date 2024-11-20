package performancetest;

import java.time.Duration;
import java.time.Instant;


public class nQueenProblem implements benchmark {

    private static final int N = 29;

    @Override
    public double runBenchmark(VirtualMachine VM) {
        System.out.println("Starting n-Queens problem performance test...");

        Instant taskStart = Instant.now();

        try {
            if (solveNQ(N)) {
                long responseTime = Duration.between(taskStart, Instant.now()).toMillis();
                System.out.println("Task completed in: " + responseTime + " ms");
                return responseTime;
            } else {
                System.out.println("No solution found.");
                return -1;
            }
        } catch (Exception e) {
            System.err.println("Error during N-Queens computation: " + e.getMessage());
            return -1;
        }
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

}

