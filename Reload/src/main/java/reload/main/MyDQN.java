package reload.main;

import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.util.DataManager;
import org.nd4j.linalg.learning.config.Adam;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import reload.config.DqnProperties;
import reload.model.QualityMeasures;
import reload.service.ILRMDP;
import reload.util.CsvWriter;

import java.io.File;
import java.io.IOException;


@SpringBootApplication
public class MyDQN {

    //configuring the neural net
    public static DQNFactoryStdDense.Configuration LOAD_TEST_NET =
            DQNFactoryStdDense.Configuration.builder()
                    .l2(0.01).updater(new Adam(1e-3)).numLayer(3).numHiddenNodes(16).build();
    public static int maxEpocStep = 30;
    public static int maxStep = 450;
    public static int expRepMaxSize = maxStep;
    public static DqnProperties dqnProperties =
            new DqnProperties(
                    123,   //random seed
                    maxEpocStep,//max step per epoch
                    maxStep, //max step, training will finish after this number of steps
                    expRepMaxSize, //max size of experience replay
                    64,    //size of batches
                    10,   //target update (hard)
                    1,     //num step noop warmup
                    0.01,  //reward scaling
                    0.5,  //gamma, discount factor
                    10.0,  //td-error clipping
                    0.1f,  //min epsilon
                    350,  //num step for eps greedy anneal
                    true   //double DQN
            );
    public static QLearning.QLConfiguration LOAD_TEST_QL =
            new QLearning.QLConfiguration(
                    dqnProperties.seed,
                    dqnProperties.maxEpochStep,
                    dqnProperties.maxStep,
                    dqnProperties.expRepMaxSize,
                    dqnProperties.batchSize,
                    dqnProperties.targetDqnUpdateFreq,
                    dqnProperties.updateStart,
                    dqnProperties.rewardFactor,
                    dqnProperties.gamma,
                    dqnProperties.errorClamp,
                    dqnProperties.minEpsilon,
                    dqnProperties.epsilonNbStep,
                    dqnProperties.doubleDQN
            );
    public static int maxResponseTimeThreshold = 200;
    public static double maxErrorRateThreshold = 0.6;
    public static int episodeExecutionDelay = 5;

    public static void loadTest() throws IOException {
        //record the training data in rl4j-data in a new folder
        DataManager manager = new DataManager();

        //logging configuration
        CsvWriter csvWriter = new CsvWriter(dqnProperties, maxResponseTimeThreshold, maxErrorRateThreshold, episodeExecutionDelay);

        //create the Intelligent Load Runner MDP
        ILRMDP mdp = new ILRMDP(maxResponseTimeThreshold, maxErrorRateThreshold, csvWriter, episodeExecutionDelay);

        //define the training method
        Learning<QualityMeasures, Integer, DiscreteSpace, IDQN> dql = new QLearningDiscreteDense<QualityMeasures>(mdp, LOAD_TEST_NET, LOAD_TEST_QL, manager);
        mdp.setFetchable(dql);
        dql.train();
        mdp.close();

    }

    public static void main(String[] args) throws IOException {
        String jmeterJarPath = System.getProperty("user.dir") + File.separator + "libs" + File.separator + "ApacheJMeter_functions-5.6.3.jar";
        System.out.println(jmeterJarPath);
        File jmeterJar = new File(jmeterJarPath);
        if (!jmeterJar.exists()) {
            System.out.println("Jmeter functions jar file not found...");
        } else {
            System.out.println("Jmeter functions jar file found at: " + jmeterJarPath);
            System.setProperty("java.class.path", System.getProperty("java.class.path") + jmeterJarPath);
        }
        SpringApplication.run(MyDQN.class, args);
        loadTest();
    }

}
