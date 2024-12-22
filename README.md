## Reinforcement_Learning_for_Software_Quality_Assurance
This repository contains the source code for a smart reinforcement learning (RL)-assisted performance testing framework. This intelligent testing framework enables efficient generation of performance test cases to meet specific testing objectives without requiring access to the source code or models of the system under test (SUT). It achieves this through three major components: SaFReL, RELOAD, and A2C.

#### Core Features
* SaFReL: Focuses on platform-based test condition generation using self-adaptive fuzzy reinforcement learning.
* RELOAD: Specializes in workload-based test condition generation using adaptive reinforcement learning.
* A2C: Implements the Advantage Actor-Critic algorithm to optimize resource configurations for stress testing. A2C focuses on determining the minimal CPU, memory, and disk utilization needed to maintain principal system functionalities while simulating resource-constrained scenarios. 

The framework enables efficient stress testing by learning optimal resource allocation policies, which can be reused in future testing scenarios. It adapts to varying resource constraints, offering a cost-effective and intelligent solution for continuous testing activities, such as system resilience testing under extreme conditions and performance evaluation under stress.

### License
Reload, Safrel_v1, Safrel_v2, A2C are open-source and distributed under:  
```
Copyright (c) 2021, mahshidhelali
Copyright (c) 2024, VervelleKillian
All rights reserved.
```

