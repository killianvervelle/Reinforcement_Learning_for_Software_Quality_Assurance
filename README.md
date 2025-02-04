## Reinforcement_Learning_for_Software_Quality_Assurance
This repository contains the source code for reinforcement learning assisted performance testing frameworks. Each framework enables efficient generation of performance test cases to meet specific testing objectives without requiring access to the source code or models of the system under test (SUT). It achieves this through three major components: SaFReL (Q-learning), RELOAD (DQN), A2C (Advantage Actor-Critic) and PPO (Proximal Policy Optimization).

#### Core Frameworks
* PPO - The primary model developed in this project and deployed in a cloud environment: Implements PPO to optimize resource configurations for stress testing.
* A2C - The secondary model developed in this project: Implements the A2C algorithm to optimize resource configurations for stress testing.
* SaFReL - Focuses on platform-based test condition generation using self-adaptive fuzzy reinforcement learning.
* RELOAD - Specializes in workload-based test condition generation using adaptive reinforcement learning.

All four enable efficient stress testing by learning optimal resource allocation policies, which can be reused in future testing scenarios. They adapt to varying resource constraints, offering a cost-effective and intelligent solution for continuous testing activities, such as system resilience testing under extreme conditions and performance evaluation under stress.

