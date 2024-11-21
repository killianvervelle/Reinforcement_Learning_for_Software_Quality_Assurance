package reload.service;

import org.apache.jmeter.control.LoopController;
import org.apache.jmeter.engine.StandardJMeterEngine;
import org.apache.jmeter.reporters.ResultCollector;
import org.apache.jmeter.reporters.Summariser;
import org.apache.jmeter.save.SaveService;
import org.apache.jmeter.threads.ThreadGroup;
import org.apache.jmeter.util.JMeterUtils;
import org.apache.jorphan.collections.HashTree;
import org.apache.jorphan.collections.SearchByClass;

import reload.model.QualityMeasures;
import reload.model.Transaction;
import reload.util.MyResultCollector;

import java.io.File;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.util.Collection;


public class LoadTester {

    private StandardJMeterEngine jmeter;
    private ResultCollector logger;
    private final String JMETER_HOME_PATH = "/home/ubuntu/apache-jmeter/";
    private final String JMETER_PROPERTY_PATH = "/home/ubuntu/apache-jmeter/bin/jmeter.properties";
    private final String JMETER_LOG_FILE_PATH = "/home/ubuntu/apache-jmeter/bin/transactions_rubis/all_transactions_local_server.jtl";
    private final String JMX_FILES_PATH = "/home/ubuntu/apache-jmeter/bin/TestUser.jmx";
    private final String[] remoteHosts = {"1XX.XX.XX.XX", "1XX.XX.X.XXX"};  //set them up here or in jmeter.properties file

    public LoadTester() {
        checkRemoteConnectivity();
        Initialize();
    }

    private void checkRemoteConnectivity() {
        System.out.println("Checking network connectivity to remote hosts...");
        for (String host : remoteHosts) {
            int rmiPort = 4000;
            System.out.println("Attempting to connect to host: " + host + " on port: " + rmiPort);
            try (Socket socket = new Socket()) {
                socket.connect(new InetSocketAddress(host, rmiPort), 3000); // 3-second timeout
                System.out.println("Successfully connected to host: " + host + " on port: " + rmiPort);
            } catch (IOException e) {
                System.out.println("Failed to connect to host: " + host + " on port: " + rmiPort);
                System.out.println("Error: " + e.getMessage());
            }
        }
    }

    private void Initialize() {
        jmeter = new StandardJMeterEngine();

        try {
            JMeterUtils.loadJMeterProperties(JMETER_PROPERTY_PATH);
            System.out.println("Loaded jmeter.properties from the path: " + JMETER_PROPERTY_PATH);
        } catch (Exception e) {
            System.out.println("Failed to load jmeter.properties from the path: " + JMETER_PROPERTY_PATH);
            e.printStackTrace();
        }

        //checking if remote hosts have been set
        String remoteHosts = JMeterUtils.getPropDefault("remote_hosts", "Not Set");
        System.out.println("remote_hosts is set to: " + remoteHosts);

        JMeterUtils.setProperty("remote_hosts", "172.31.10.74,172.31.4.179");
        JMeterUtils.setJMeterHome(JMETER_HOME_PATH);

        try {
            JMeterUtils.initLocale();
            System.out.println("Waiting for remote nodes to be ready...");
            Thread.sleep(5000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            e.printStackTrace();
        }

        try {
            SaveService.loadProperties();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public boolean ExecuteAllTransactions(Transaction[] transactions, int rampUpTime, int numOfLoops, QualityMeasures qualityMeasures) {
        Summariser summer = null;
        String summariserName = JMeterUtils.getPropDefault("summariser.name", "summary");
        if (!summariserName.isEmpty()) {
            summer = new Summariser(summariserName);
        }

        String logFile = JMETER_LOG_FILE_PATH;
        File logFileDir = new File(logFile).getParentFile();
        if (!logFileDir.exists()) {
            logFileDir.mkdirs();
        }
        logger = new ResultCollector(summer);
        logger.setFilename(logFile);

        HashTree testPlanTree;
        boolean testPassed = false;
        String transactionName = "all_transactions";
        try {
            testPlanTree = SaveService.loadTree(new File(JMX_FILES_PATH));
            System.out.println("Test plan tree:" + testPlanTree.toString());

            SearchByClass<ThreadGroup> threadGroups = new SearchByClass<>(ThreadGroup.class);
            testPlanTree.traverse(threadGroups);
            Collection<ThreadGroup> threadGroupsRes = threadGroups.getSearchResults();
            for (ThreadGroup threadGroup : threadGroupsRes) {
                for (Transaction t : transactions) {
                    if ((t.name).equals(threadGroup.getName())) {
                        threadGroup.setNumThreads(t.workLoad);
                        threadGroup.setRampUp(rampUpTime);
                        ((LoopController) threadGroup.getSamplerController()).setLoops(numOfLoops);

                        System.out.println("thread group: " + threadGroup.getName());
                        System.out.println("Transaction: " + t.name + ", Workload (num of threads): " + threadGroup.getProperty("ThreadGroup.num_threads").toString());
                        System.out.println("Ramp Up:" + threadGroup.getRampUp());
                        System.out.println("Loop: " + threadGroup.getSamplerController().getProperty("LoopController.loops"));
                        break;
                    }
                }
            }
            System.out.println("Ramp Up:" + rampUpTime);

            MyResultCollector myResultCollector = new MyResultCollector(summer);
            testPlanTree.add(testPlanTree.getArray()[0], myResultCollector);
            // Run JMeter Test
            jmeter.configure(testPlanTree);

            try {
                jmeter.run();
                testPassed = myResultCollector.allTestSamplesPassed();
                if (testPassed)
                    myResultCollector.calculateAverageQualityMeasures(qualityMeasures);
            } catch (Exception e) {
                System.out.println("jmeter run failed.");
                e.printStackTrace();
            }

        } catch (Exception e) {
            System.out.println("testPlanTree failed.");
            e.printStackTrace();
        }

        return testPassed;
    }
}
