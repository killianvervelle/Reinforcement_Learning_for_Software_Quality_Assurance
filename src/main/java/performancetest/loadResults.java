package performancetest;

/**
 ** @author vervellekillian
 **/

public class loadResults {
    public final int successfullRequests;
    public final double averageResponseTime;

    public loadResults(int successfullRequests, double averageResponseTime) {
        this.averageResponseTime = averageResponseTime;
        this.successfullRequests = successfullRequests;
    }

    public int getSuccessfulRequests() {
        return successfullRequests;
    }

    public double getAverageResponseTime() {
        return averageResponseTime;
    }
}