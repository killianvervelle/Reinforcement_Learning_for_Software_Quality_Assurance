package reload.model;

import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


public final class SUTObservationSpace implements ObservationSpace<QualityMeasures> {

    public int maxResponseTimeThreshold;
    public double maxErrorRateThreshold;

    public SUTObservationSpace(int maxResponseTimeThreshold, double maxErrorRateThreshold) {
        this.maxResponseTimeThreshold = maxResponseTimeThreshold;
        this.maxErrorRateThreshold = maxErrorRateThreshold;
    }

    @Override
    public String getName() {
        return "Quality Measures";
    }

    @Override
    public int[] getShape() {
        return new int[]{2};
    }

    @Override
    public INDArray getLow() {
        INDArray low = Nd4j.create(new float[]{0, 0});
        return low;
    }

    @Override
    public INDArray getHigh() {
        INDArray high = Nd4j.create(new float[]{maxResponseTimeThreshold, (float) maxErrorRateThreshold});
        return high;
    }
}