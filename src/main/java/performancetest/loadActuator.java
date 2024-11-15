package performancetest;

import java.net.URI;
import java.net.http.*;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 ** @author vervellekillian
 **/

public class loadActuator {

    private static final String TOKEN_ISS_REQ = "http://localhost:9100/api/v1/issuer/issue";
    // private static final String TOKEN_TRANSF_REQ = "POST http://localhost:9200/api/v1/owner/accounts/alice/transfer";
    private static final int TEST_DURATION_SECONDS  = 120; //in seconds

    private List<Long> responseTimes = Collections.synchronizedList(new ArrayList<>());
    private AtomicInteger successfullRequests = new AtomicInteger(0);

    private void loadRequests() throws InterruptedException {
        ExecutorService executorService = Executors.newCachedThreadPool();
        Instant startTime = Instant.now();

        while (Duration.between(startTime, Instant.now()).getSeconds() < TEST_DURATION_SECONDS) {
            issueTokens(executorService);
        }
        executorService.shutdown();
        double averageResponseTime = responseTimes.stream().mapToLong(Long::longValue).average().orElse(0);

        System.out.println("Throughput: " + successfullRequests);
        System.out.println("Average response time: " + averageResponseTime + " ms");
    }

    private void issueTokens(ExecutorService executorService) {
        HttpClient client = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_2)
                .connectTimeout(Duration.ofSeconds(10))
                .build();

        String jsonContent = "{\n" +
                "  \"amount\": {\"code\": \"TOK\", \"value\": 100},\n" +
                "  \"counterparty\": {\"node\": \"owner1\", \"account\": \"alice\"},\n" +
                "  \"message\": \"Success\"\n" +
                "}";

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(TOKEN_ISS_REQ))
                .timeout(Duration.ofMinutes(2))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(jsonContent))
                .build();

        long requestStartTime = System.nanoTime();

        CompletableFuture<Void> future = client.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(HttpResponse::body)
                .thenAccept(responseBody -> {
                    long totalTime = (System.nanoTime() - requestStartTime) / 1_000_000; // in ms
                    responseTimes.add(totalTime);
                    successfullRequests.incrementAndGet();
                    System.out.println("Request successful, Response time: " + totalTime + " ms");
                })
                .exceptionally(ex -> {
                    System.out.println("Request failed: " + ex.getMessage());
                    return null;
                });

        executorService.submit(future::join);
    }
}
