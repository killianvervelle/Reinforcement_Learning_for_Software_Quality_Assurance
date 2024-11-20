package reload.model;

import org.deeplearning4j.rl4j.space.Encodable;


public abstract class QualityMeasures implements Encodable {

    public double responseTime;
    public double errorRate;

    public QualityMeasures(double responseTime, double errorRate) {
        this.responseTime = responseTime;
        this.errorRate = errorRate;
    }

    public void update(double responseTime, double errorRate) {
        if (responseTime < 0 || errorRate < 0) {
            throw new IllegalArgumentException("Response time and error rate must be non-negative.");
        }
        this.responseTime = responseTime;
        this.errorRate = errorRate;
    }

    @Override
    public String toString() {
        return String.format("responseTime: %.2f\nerrorRate: %.2f", responseTime, errorRate);
    }

    @Override
    public double[] toArray() {
        return new double[]{this.responseTime, this.errorRate};
    }

}
