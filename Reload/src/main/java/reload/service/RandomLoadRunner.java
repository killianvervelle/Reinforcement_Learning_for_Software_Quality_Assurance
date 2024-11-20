package reload.service;

import reload.model.SUTstate;
import reload.util.CsvWriter;

import java.util.Random;
import java.util.concurrent.TimeUnit;


public class RandomLoadRunner {

    private SUT sutEnv;
    private SUTstate currSUTState;
    private int iterationNumber = 1;
    private int iterationStep = 1;
    private final Random random;
    private final int maxResponseTimeThreshold;
    private final double maxErrorRateThreshold;
    private final CsvWriter csvWriter;
    private final int episodeExecutionDelay;
    private final int maxStep;

    private static final int MAX_ACTIONS = 12;

    public RandomLoadRunner(int maxResponseTimeThreshold, double maxErrorRateThreshold, CsvWriter csvWriter, int episodeExecutionDelay, int maxStep) {
        this.sutEnv = new SUT();
        this.random = new Random();
        this.maxResponseTimeThreshold = maxResponseTimeThreshold;
        this.maxErrorRateThreshold = maxErrorRateThreshold;
        this.csvWriter = csvWriter;
        this.episodeExecutionDelay = episodeExecutionDelay;
        this.maxStep = maxStep;
    }

    public void execute(int maxIterationNumber) {
        int currStep = 1;
        int action;

        while (iterationNumber <= maxIterationNumber) {
            //this.maxResposeTimeThreshold += 100;
            //this.maxErrorRateThreshold += 0.01;
            this.currSUTState = this.sutEnv.getSUTState();

            while (!isDone()) {
                action = getRandomAction();
                this.sutEnv.applyAction(action);
                this.currSUTState = this.sutEnv.getSUTState();

                logStatus(this.sutEnv.transactions[action].name);
                iterationStep = iterationStep + 1;

            }
            currStep = currStep + iterationStep;
            iterationStep = 1;
            iterationNumber = iterationNumber + 1;

            try {
                TimeUnit.MINUTES.sleep(episodeExecutionDelay);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    public boolean isDone() {
        boolean done = false;
        if (this.currSUTState.qualityMeasures.errorRate > maxErrorRateThreshold || this.currSUTState.qualityMeasures.responseTime > maxResponseTimeThreshold) {
            done = true;
            csvWriter.writeRowInLog(iterationNumber, iterationStep - 1, this.currSUTState);
            System.out.println("The process has ended.");
        }

        return done;
    }

    public int getRandomAction() {
        return random.nextInt(MAX_ACTIONS);
    }

    public void logStatus(String transactionName) {
        System.out.println("iteration: " + iterationNumber + ", iterationStep: " + iterationStep);
        System.out.println(this.currSUTState.toString());

        csvWriter.writeRowInFullLog(iterationNumber, iterationStep, transactionName, this.currSUTState);
    }

}
