### A2C Framework: Advantage Actor-Critic for Reinforcement Learning
The A2C Framework implements the Advantage Actor-Critic algorithm, a reinforcement learning approach designed to balance exploration and exploitation. The algorithm utilizes two neural networks: an actor to determine actions and a critic to estimate value functions, optimizing the agent's performance in a custom environment. 
<div align="center">
  <img src="../img/a2c.png" alt="Alt text" width="500">
</div>

#### Key Features:
- Implements the Advantage Actor-Critic algorithm for reinforcement learning.
- Trains agents in a custom-defined environment.
- Uses neural networks for policy optimization and value estimation.
- Supports dynamic adjustments to the agent's learning process through tunable hyperparameters.

#### How it works:
The A2C Framework trains agents to optimize rewards in a reinforcement learning environment. The actor network selects actions based on current states, while the critic network evaluates those actions to guide the actor toward better decisions. 

#### Reference: 
For detailed insights into the A2C algorithm and its implementation, refer to the foundational research:
https://paperswithcode.com/method/a2c

#### Setting Up the Framework
#### Prerequisites
* Python: Version 3.10 or higher.
* Dependencies: Install the required libraries.
* Hardware: A GPU is recommended for faster training but is not required.


#### Running SaFReL_v1 or SaFReL_v2

Clone the repository locally:
```
git clone https://github.com/killianvervelle/Reinforcement_Learning_for_Software_Quality_Assurance
```
Set Up the Environment:
```
cd Reinforcement_Learning_for_Software_Quality_Assurance/a2c
pip install -r requirements.txt
```
Run the A2C Framework:
```
python src/main.py
```

#### Visualization of Results
Graphs and metrics summarizing the agent's performance can be visualized using TensorBoard or the custom plotting script in the main file:
<div align="center">
  <img src="../img/a2cresults.png" alt="Alt text" width="2000">
</div>