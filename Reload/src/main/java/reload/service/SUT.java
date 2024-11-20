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

            @Override
            public boolean isSkipped() {
                return false;
            }

            @Override
            public INDArray getData() {
                return Nd4j.createFromArray(toArray());
            }

            @Override
            public Encodable dup() {
                return null;
            }
        };

        transactions[0] = new Transaction("Test1", initialWorkLoadPerTransaction);
        transactions[1] = new Transaction("RegisterPage", initialWorkLoadPerTransaction);
        transactions[2] = new Transaction("RegisterUser", initialWorkLoadPerTransaction);
        transactions[3] = new Transaction("BrowsePage", initialWorkLoadPerTransaction);
        transactions[4] = new Transaction("BrowseInCategory", initialWorkLoadPerTransaction);
        transactions[5] = new Transaction("BrowseInRegion", initialWorkLoadPerTransaction);
        transactions[6] = new Transaction("SellPage", initialWorkLoadPerTransaction);
        transactions[7] = new Transaction("SellItem", initialWorkLoadPerTransaction);
        transactions[8] = new Transaction("AboutMePage", initialWorkLoadPerTransaction);
        transactions[9] = new Transaction("AboutMeUser", initialWorkLoadPerTransaction);
        transactions[10] = new Transaction("BidOnItem", initialWorkLoadPerTransaction);
        transactions[11] = new Transaction("SellItem", initialWorkLoadPerTransaction);

    }

    public void applyAction_base() {
        boolean success = false;
        while (!success) {
            try {
                success = executeTestPlan();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
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


