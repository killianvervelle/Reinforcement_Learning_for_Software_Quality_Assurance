### RELOAD: Adaptive RL-Driven Load Testing
RELOAD is a test agent that efficiently generates and executes 
workload-based performance test cases on the SUT. Using 
reinforcement learning, RELOAD learns the effects of different 
transactions and their optimal load configurations to meet specific 
testing objectives, such as achieving desired response times or error 
rates.
<div align="center">
  <img src="../img/reload.png" alt="Alt text" width="550">
</div>

#### Key Features:
- Generates effective workloads with minimal cost;
- Utilizes Apache JMeter for workload execution;
- Learns and reuses optimal workload generation policies.

#### How it Works:
RELOAD learns to tune the transaction loads in the workload to 
achieve test objectives. This intelligent agent is particularly 
effective for continuous testing scenarios, such as:
- Testing varying performance conditions;
- Performance regression testing; 
- Testing under dynamic workload requirements.

<div align="center">
  <img src="../img/distributed.png" alt="Alt text" width="450">
</div>

#### Reference: 
For detailed insights into Reload’s structure and mechanism, refer to the published work:
https://arxiv.org/pdf/2104.12893.

For details on how to implement Distributed Testing, refer to:
https://jmeter.apache.org/usermanual/jmeter_distributed_testing_step_by_step.html

For details on how to build or record a testing plan, refer to:
https://jmeter.apache.org/usermanual/jmeter_proxy_step_by_step.html

For the official implementation of Apache Jmeter, refer to:
https://jmeter.apache.org/

#### Setting Up the Framework
#### Prerequisites
* Java: Version 17 or higher.
* Maven: For dependency management and building the project.
* Apache JMeter: Installed and configured for executing performance tests.
* Hardware Requirements: Ensure sufficient system resources for CPU, memory, and disk-intensive testing scenarios.

#### Running Reload

Set up your cloud environment (here AWS). Launch 3 EC2 instances (1 controller, 2 workers)
- Instance Type: t2.medium
- OS: Ubuntu 22.4 
- Number of vCPUs: 2
- Volume size: 16GB

Set up the security group for your controller. Allow inbound connections on ports 1099, 50000, and 22:
<div align="center">
  <img src="../img/controller-SG.png" alt="Alt text" width="550">
</div>

Set up the security group for your workers. Allow inbound connections on ports 4000, 1099, and 22:
<div align="center">
  <img src="../img/worker-SG.png" alt="Alt text" width="550">
</div>

On each node, install and set up Apache Jmeter and Distributed Testing:
```
wget https://dlcdn.apache.org//jmeter/binaries/apache-jmeter-5.6.3.tgz
tar -xvzf apache-jmeter-5.6.3.tgz
export JMETER_HOME=/path/to/apache-jmeter
export PATH=$JMETER_HOME/bin:$PATH
```
Verify the installation:
```
jmeter -v
```
Design a test plan manually or by using JMeter's Recorder JMeter proxy: [JMeter proxy](https://jmeter.apache.org/usermanual/jmeter_proxy_step_by_step.html).

On the controller node, edit the jmeter.properties file located in the bin directory:
```
cd apache-jmeter-5.6.3/bin
nano jmeter-server.properties
```
- remote_hosts=\<worker1-private-IP\>\, \<worker2-private-IP\>
- Set client.rmi.localport=50000  
- Set client.tries=3  
- Set client.retries_delay=1000
- Set server.rmi.ssl.disable=true

On the worker nodes, edit the jmeter.properties file located in the bin directory:
```
cd apache-jmeter-5.6.3/bin
nano jmeter-server.properties
```
- Set server.rmi.localport=4000  
- Set server.rmi.ssl.disable=true

```
# Clone the repository locally
git clone https://github.com/killianvervelle/Reinforcement_Learning_for_Software_Quality_Assurance
```
In the class LoadTester.java, update the following variables. Paths will be different if implementation done locally or in cloud instances.
- JMETER_HOME_PATH = <JMETER_HOME>
- JMETER_PROPERTY_PATH = <JMETER_HOME>/bin/jmeter.properties 
- JMETER_LOG_FILE_PATH = <JMETER_HOME>/bin/transactions_rubis/all_transactions_local_server.jtl 
- JMX_FILES_PATH = <JMETER_HOME>/bin/your-test-plan.jmx

Build the project using Maven and install all dependencies
```
cd Reload
mvn clean install
```

Send the project.jar to your controller node. Use the same key pair assigned at launch of your instance.
```
scp -i <key.pem> <path-to-project-jar> ubuntu@<controller-public-IP>:<JMETER_HOME>/bin/
```


Ensure that all remote nodes have access to the required test plan and dependencies
On each worker node, start the JMeter server.
```
./jmeter-server
```

On the controller node:
If you are using a pre-defined JMX test plan for distributed testing:
```
jmeter -n -t <test-plan-path>.jmx -r
```

For running custom logic with RELOAD:
```
java -jar <path-to-project-jar>
```

### License
Reload and Safrel are open-source and distributed under:  
```
Copyright (c) 2021, mahshidhelali
All rights reserved.
```