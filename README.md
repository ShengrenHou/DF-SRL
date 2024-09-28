# DistFlow Safe Reinforcement Learning Algorithm for Voltage Magnitude Regulation in Distribution Networks

Source code for the paper: DistFlow Safe Reinforcement Learning Algorithm for Voltage Magnitude Regulation in Distribution Networks.

# Abstract 
The integration of distributed energy resources (DER) has escalated the challenge of voltage magnitude regulation in distribution networks. Model-based approaches, which rely on complex sequential mathematical formulations, can not meet real-time demand. Deep reinforcement learning (DRL) offers an alternative by utilizing offline training with distribution network simulators and then execution without online computation. However, DRL algorithms fail to enforce voltage magnitude constraints during training and testing, potentially leading to serious operational violations. To tackle these challenges, we introduce a novel safe reinforcement learning algorithm, the DistFlow Safe Reinforcement Learning (DF-SRL), designed specifically for real-time voltage magnitude regulation in distribution networks. The DF-SRL algorithm incorporates a DistFlow linearization to construct an expert knowledge-based safety layer. Subsequently, DF-SRL overlays this safety layer on top of the agent's policy, recalibrating unsafe actions to safe domains through a quadratic programming formulation. Simulation results show the proposed DF-SRL consistently ensures voltage magnitude constraints during the training and real-time operation (test) phases, achieving faster convergence and higher performance, setting it apart from (safe) DRL benchmarks.

# Organization
* Folder "data_sources" -- Historical and processed data for 18 nodes network.
* script "DF-SRL" -- The algorithm we developed.
* script "safe_battery_env"-- General environment and the safe layer we developed 
* Run scripts DF-SRL.py after installing all packages. Please have a look for the code structure.

# Recommended citation
Hou S, Fu A, Duque E M S, et al. DistFlow Safe Reinforcement Learning Algorithm for Voltage Magnitude Regulation in Distribution Networks[J]. Journal of Modern Power Systems and Clean Energy, 2024.
