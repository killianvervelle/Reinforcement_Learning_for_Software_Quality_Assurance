package com.example.performancetest;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import performancetest.PerformanceTest;

@SpringBootTest(classes = PerformanceTest.class)
class PerformanceTests {
    @Test
    void contextLoads() {
    }
}