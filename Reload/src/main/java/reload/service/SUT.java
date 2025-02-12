package reload.service;

import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import reload.model.QualityMeasures;
import reload.model.SUTstate;
import reload.model.Transaction;


public class SUT {

    public LoadTester loadTester;
    public Transaction[] transactions;
    public QualityMeasures qualityMeasures;
    private double threadPerSecond;
    private int initialWorkLoadPerTransaction;
    private double workLoadIncreasingStepRatio;

    public SUT() {Initialize();
    }

    public void Initialize() {

        this.threadPerSecond = 10.00;
        this.initialWorkLoadPerTransaction = 100;
        this.workLoadIncreasingStepRatio = 4.0;

        loadTester = new LoadTester();
        transactions = new Transaction[12];
        qualityMeasures = new QualityMeasures(0, 0) {


        transactions[0] = new Transaction("Test1", initialWorkLoadPerTransaction);
        transactions[1] = new Transaction("RegisterPage", initialWorkLoadPerTransaction);
        transactions[2] = new Transaction("RegisterUser", initialWorkLoadPerTransaction);
                .......

    }

    public void applyAction(int action) {  
        boolean success = false;
        //modifying the load of transactions
        int prevWorkLoad = transactions[action].workLoad;
        transactions[action].workLoad = (int) (prevWorkLoad * workLoadIncreasingStepRatio);

        while (!success) {
            try {
                success = executeTestPlan();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    public boolean executeTestPlan() {
        boolean success = false;
        int rampUpTime = (int) Math.round((double) GetTotalWorkLoad() / threadPerSecond);

        try {
            success = loadTester.ExecuteAllTransactions(transactions, rampUpTime, 1, qualityMeasures);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return success;
    }

    public int GetTotalWorkLoad() {
        int totalWorkLoad = 0;
        for (Transaction t : transactions) totalWorkLoad += t.workLoad;
        return totalWorkLoad;
    }

    public int GetInitialTotalWorkLoad() {
        return initialWorkLoadPerTransaction * transactions.length;
    }

    public SUTstate getSUTState() {
        return new SUTstate(qualityMeasures, GetTotalWorkLoad(), transactions);
    }
    
}


