## Reinforcement_Learning_for_Software_Quality_Assurance
This repository contains the source code for a smart reinforcement learning (RL)-assisted performance testing framework. This intelligent testing framework enables efficient generation of performance test cases to meet specific testing objectives without requiring access to the source code or models of the system under test (SUT). It achieves this through two major components: SaFReL and RELOAD.
#### Core Features
* SaFReL: Focuses on platform-based test condition generation using self-adaptive fuzzy reinforcement learning.
* RELOAD: Specializes in workload-based test condition generation using adaptive reinforcement learning.

The framework supports efficient performance testing by learning optimal test case generation policies, which can then be reused in further testing scenarios. It adapts to varying test conditions, offering a cost-effective and intelligent solution for continuous testing activities such as performance regression testing.

### Project structure
parent-project/
│
├── Safrel_v1/          
│   ├── src/main/java/  
│   ├── pom.xml         
│
├── Safrel_v2/         
│   ├── src/main/java/ 
│   ├── pom.xml         
│
└── README.md  



### SaFReL: Self-Adaptive Fuzzy Reinforcement Learning
SaFReL is a performance testing agent designed to generate platform-based test cases. It learns 
to optimally tune resource availability (CPU, memory, disk) to 
identify the performance breaking point for various SUTs. The 
framework uses a performance estimation module to predict the 
behavior of software programs based on their sensitivity to 
hardware resources.
<div align="center">
  <img src="./img/Safrel.png" alt="Alt text" width="600">
</div>

#### Key Features:
- Learns and replays policies to generate platform-based performance 
test cases. 
- Works on CPU-intensive, memory-intensive, and disk-intensive SUTs. 
- Efficiently identifies resource configurations to meet testing objectives.

#### How it works:
SaFReL tunes hardware configurations to simulate resource-constrained scenarios, 
enabling a thorough evaluation of the SUT's performance. The learned policies can be applied 
to new testing cases, making the framework adaptive and reusable.

#### Reference: 
For detailed insights into SaFReL’s structure and mechanism, refer to the published work:
https://link.springer.com/article/10.1007/s11219-020-09532-z

#### Setting Up the Framework
##### Prerequisites
* Java: Version 17 or higher. 
* Maven: For dependency management and building the project. 
* Hardware Requirements: Ensure sufficient system resources for CPU, memory, and disk-intensive testing scenarios.

##### Installation
1) Clone the repository

2) Build the project using Maven and install all dependencies

3) Update configurations of the respective config files as per your testing environment


#### Running SaFReL_v1 or SaFReL_v2
```
git clone https://github.com/killianvervelle/Reinforcement_Learning_for_Software_Quality_Assurance
cd Safrel/Safrel_v1
mvn clean install
java -jar target/my-Safrel-v1.jar
```

### RELOAD: Adaptive RL-Driven Load Testing
RELOAD is a test agent that efficiently generates and executes 
workload-based performance test cases on the SUT. Using 
reinforcement learning, RELOAD learns the effects of different 
transactions and their optimal load configurations to meet specific 
testing objectives, such as achieving desired response times or error 
rates.
<div align="center">
  <img src="./img/reload.png" alt="Alt text" width="550">
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

#### Reference: 
For detailed insights into Reload’s structure and mechanism, refer to the published work:
https://arxiv.org/pdf/2104.12893

#### Setting Up the Framework
##### Prerequisites
* Java: Version 17 or higher.
* Maven: For dependency management and building the project.
* Apache JMeter: Installed and configured for executing performance tests.
* Hardware Requirements: Ensure sufficient system resources for CPU, memory, and disk-intensive testing scenarios.

##### Installation
1) Clone the repository

2) Build the project using Maven and install all dependencies

3) Set up Apache JMeter

4) Update configurations of the respective config files as per your testing environment


#### Running Reload
```
# Clone the repository 
git clone https://github.com/killianvervelle/Reinforcement_Learning_for_Software_Quality_Assurance

# Set up Apach Jmeter and Distributed Testing
tar -xvzf https://dlcdn.apache.org//jmeter/binaries/apache-jmeter-5.6.3.tgz
export JMETER_HOME=/path/to/apache-jmeter
export PATH=$JMETER_HOME/bin:$PATH
# Verify installation
jmeter -v

# Configure JMeter for Distributed Testing in AWS
# Launch 3 ec2 instances (1 controller, 2 workers) 
Instance type: t2.medium
Platform: Ubuntu 22.4 
Number of vCPUs: 2
Volume size (GB): 16

# Edit the jmeter.properties file located in the bin directory
cd apache-jmeter-5.6.3/bin
nano jmeter-server.properties
# Set remote_hosts=XXX.XXX.X.XXX,XXX.XXX.X.XXX (IPv4 adresses of your worker nodes)
# On each worker node, start the JMeter server
./jmeter-server
# On the controller node, execute the test plan with the following command to use the worker nodes
jmeter -n -t nameOfYourTestPlan.jmx -r
# Ensure that all remote nodes have access to the required test plan and dependencies

# Go to and Build the project using Maven and install all dependencies
cd Reload
mvn clean install

# Run the project
java -jar target/my-reload.jar
```

### License
Reload, Safrel_v1 and Safrel_v2 are open-source and distributed under:  
```
Copyright (c) 2021, mahshidhelali
Copyright (c) 2024, VervelleKillian
All rights reserved.
```

