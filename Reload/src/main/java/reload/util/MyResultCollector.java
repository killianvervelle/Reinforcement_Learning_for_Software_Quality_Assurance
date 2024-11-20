package reload.util;

import org.apache.jmeter.reporters.ResultCollector;
import org.apache.jmeter.reporters.Summariser;
import org.apache.jmeter.samplers.SampleEvent;
import org.apache.jmeter.samplers.SampleResult;

import reload.model.QualityMeasures;


public class MyResultCollector extends ResultCollector {

    private long responseTime;
    private int errorCount;
    private int numOfSamples;


    public MyResultCollector(Summariser summer) {
        super(summer);
        responseTime = 0;
        errorCount = 0;
        numOfSamples = 0;
    }

    @Override
    public void sampleOccurred(SampleEvent e) {
        super.sampleOccurred(e);
        SampleResult r = e.getResult();
        if (r.isSuccessful()) {
            responseTime = responseTime + r.getTime();
        }

        errorCount = errorCount + r.getErrorCount();
        numOfSamples = numOfSamples + r.getSampleCount();
    }

    public void calculateAverageQualityMeasures(QualityMeasures QM) {
        double avgResponseTime = responseTime / (double) numOfSamples;
        double avgErrorRate = errorCount / (double) numOfSamples;
        System.out.println("Average Response Time in milliseconds: " + avgResponseTime);
        System.out.println("Error Rate: " + avgErrorRate);

        QM.update(avgResponseTime, avgErrorRate);
    }

    public boolean allTestSamplesPassed() {
        return numOfSamples != 0;
    }

}
