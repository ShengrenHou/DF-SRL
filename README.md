# DistFlow Safe Reinforcement Learning Algorithm for Voltage Magnitude Regulation in Distribution Networks

Source code for the paper: DistFlow Safe Reinforcement Learning Algorithm for Voltage Magnitude Regulation in Distribution Networks.

# Abstract 
The integration of distributed energy resources (DERs) has escalated the challenge of voltage magnitude regulation in distribution networks. Model-based approaches, which rely on complex sequential mathematical formulations, cannot meet the real-time demand. Deep reinforcement learning (DRL) offers an alternative by utilizing offline training with distribution network simulators and then executing online without computation. However, DRL algorithms fail to enforce voltage magnitude constraints during training and testing, potentially leading to serious operational violations. To tackle these challenges, we introduce a novel safe-guaranteed reinforcement learning algorithm, the DistFlow safe reinforcement learning (DF-SRL), designed specifically for real-time voltage magnitude regulation in distribution networks. The DF-SRL algorithm incorporates a DistFlow linearization to construct an expert-knowledge-based safety layer. Subsequently, the DF-SRL algorithm overlays this safety layer on top of the agent policy, recalibrating unsafe actions to safe domains through a quadratic programming formulation. Simulation results show the DF-SRL algorithm consistently ensures voltage magnitude constraints during training and real-time operation (test) phases, achieving faster convergence and higher performance, which differentiates it apart from (safe) DRL benchmark algorithms.

# Repository Structure
* `data_sources/` - Historical and processed data for 18 nodes distribution network
* `DF-SRL.py` - Implementation of the DistFlow Safe Reinforcement Learning algorithm
* `safe_battery_env.py` - Environment implementation with the safety layer based on DistFlow linearization
* `network.py` - Network configuration and utility functions
* `requirements.txt` - Required packages for running the code

# Installation
1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

# Usage
Run the main script to train and test the DF-SRL algorithm:
```
python DF-SRL.py
```

# Algorithm Overview
The DF-SRL algorithm consists of two main components:
1. A deep reinforcement learning agent (TD3-based) for voltage regulation
2. A safety layer based on DistFlow linearization that ensures voltage constraints are satisfied

The safety layer recalibrates potentially unsafe actions to safe domains through quadratic programming, ensuring that voltage magnitude constraints are consistently maintained during both training and operation.

# Recommended Citation
Hou S, Fu A, Duque E M S, et al. DistFlow Safe Reinforcement Learning Algorithm for Voltage Magnitude Regulation in Distribution Networks[J]. Journal of Modern Power Systems and Clean Energy, 2024.
