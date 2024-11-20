package reload.service;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.NeuralNetFetchable;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.json.JSONException;
import org.json.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import reload.model.QualityMeasures;
import reload.model.SUTObservationSpace;
import reload.model.SUTstate;
import reload.util.CsvWriter;

import java.time.LocalTime;
import java.util.concurrent.TimeUnit;


public class ILRMDP implements MDP<QualityMeasures, Integer, DiscreteSpace> {

    DiscreteSpace actionSpace = new DiscreteSpace(12);
    SUTObservationSpace observationSpace;
    SUTstate curr_SUT_state;
    SUTstate prev_SUT_state;
    int maxResposeTimeThreshold;
    double maxErrorRateThreshold;
    int episodeExecutionDelay;
    int episodeNumber;
    int episodeStep;
    CsvWriter csvWriter;
    SUT SUT_env;
    NeuralNetFetchable<IDQN> fetchable;


    public ILRMDP(int maxResposeTimeThreshold, double maxErrorRateThreshold, CsvWriter csvWriter, int episodeExecutionDelay) {
        SUT_env = new SUT();
        actionSpace = new DiscreteSpace(actionSpace.getSize());
        observationSpace = new SUTObservationSpace(maxResposeTimeThreshold, maxErrorRateThreshold);
        curr_SUT_state = SUT_env.getSUTState();
        episodeNumber = 0;
        episodeStep = 0;

        this.maxResposeTimeThreshold = maxResposeTimeThreshold;
        this.maxErrorRateThreshold = maxErrorRateThreshold;
        this.episodeExecutionDelay = episodeExecutionDelay;
        this.csvWriter = csvWriter;
    }

    @Override
    public ObservationSpace<QualityMeasures> getObservationSpace() {
        return this.observationSpace;
    }

    @Override
    public DiscreteSpace getActionSpace() {
        return this.actionSpace;
    }

    @Override
    public QualityMeasures reset() {
        csvWriter.writeRowInLog(episodeNumber, episodeStep - 1, curr_SUT_state);

        try {
            TimeUnit.MINUTES.sleep(episodeExecutionDelay);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        episodeStep = 1; //resetting episode step
        episodeNumber = episodeNumber + 1;
        SUT_env = new SUT();
        SUT_env.applyAction_base(); //initializing the first step of the episode
        curr_SUT_state = SUT_env.getSUTState();
        return curr_SUT_state.qualityMeasures;
    }

    @Override
    public void close() {
    }

    @Override
    public StepReply<QualityMeasures> step(Integer action) {
        if (this.fetchable != null) {
            INDArray input = Nd4j.create(1, (curr_SUT_state.qualityMeasures).toArray().length);
            input.putRow(1, Nd4j.create((curr_SUT_state.qualityMeasures).toArray()));

            INDArray output = this.fetchable.getNeuralNet().output(input);
            System.out.println(output.toString());
        }

        prev_SUT_state = curr_SUT_state;
        LocalTime startTime = LocalTime.now();
        SUT_env.applyAction(action);
        LocalTime endTime = LocalTime.now();
        curr_SUT_state = SUT_env.getSUTState();

        String transactionName = SUT_env.transactions[action].name;
        logStatus(transactionName, startTime, endTime);
        episodeStep = episodeStep + 1;

        double reward = CalculateReward();

        try {
            return new StepReply<>(curr_SUT_state.qualityMeasures, reward, this.isDone(), new JSONObject("{\"transaction\":\"" + transactionName + "\"}"));
        } catch (JSONException e) {
            throw new RuntimeException(e);
        }
    }

    private double CalculateReward() {
        double normalizedResponseTime = curr_SUT_state.qualityMeasures.responseTime * 100 / maxResposeTimeThreshold;
        double normalizedErrorRate = curr_SUT_state.qualityMeasures.errorRate * 100 / maxErrorRateThreshold;
        return Math.pow(normalizedResponseTime, 2) + Math.pow(normalizedErrorRate, 2);

    }

    @Override
    public boolean isDone() {
        boolean done = false;
        if (curr_SUT_state.qualityMeasures.errorRate > maxErrorRateThreshold || curr_SUT_state.qualityMeasures.responseTime > maxResposeTimeThreshold) {
            done = true;
            //this.logger.info("Mission ended");
            System.out.println("Mission ended");
        }
        return done;
    }

    @Override
    public MDP<QualityMeasures, Integer, DiscreteSpace> newInstance() {
        return null;
    }

    //used for logging the output of the ANN which is the Qvalues of all actions of current state
    public void setFetchable(NeuralNetFetchable<IDQN> fetchable) {
        this.fetchable = fetchable;
    }

    public void logStatus(String transactionName, LocalTime startTime, LocalTime endTime) {
        System.out.println("episode: " + episodeNumber + ", episodeStep: " + episodeStep);
        System.out.println("StepReply: " + transactionName);
        System.out.println(curr_SUT_state.toString());

        csvWriter.writeRowInFullLog(episodeNumber, episodeStep, transactionName, curr_SUT_state, startTime, endTime);
    }

}


