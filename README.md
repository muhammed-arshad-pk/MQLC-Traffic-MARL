# MQLC: Collaborative Multi-Agent Q-Learning for Autonomous Lane-Changing

## Overview

This repository implements **MQLC (Multi-Agent Q-Learning with Collaboration)**, a research-oriented Multi-Agent Reinforcement Learning (MARL) framework for safe and efficient autonomous lane-changing in mixed traffic environments containing Autonomous Vehicles (AVs) and Human-Driven Vehicles (HDVs).

The framework follows the **Centralized Training with Decentralized Execution (CTDE)** paradigm and integrates:

- Individual decision-making using Deep Q-Networks  
- Global coordination via a centralized joint-action Q-Network  
- Graph Convolutional Networks (GCN) for inter-vehicle interaction modeling  
- Urgency-aware reward shaping for collision reduction and traffic efficiency  

The system is evaluated in **highway-env** under **sparse, normal, and dense traffic**, with up to **5 cooperative AV agents and 30 surrounding vehicles**.

---

## Key Contributions & Quantitative Results

### 1. Hybrid Dual Q-Network Architecture
- Individual Q-Network for per-agent lane-change and speed actions  
- Global Joint Q-Network for coordinated multi-agent planning  
- Consistency-regularized loss to align local and global value functions  

### 2. Graph-Based Interaction Modeling
- Vehicles modeled as nodes with proximity-based edges  
- **GCN-GRU** captures spatio-temporal interaction and maneuver intent  
- Enables anticipation of cut-ins, merges, and cooperative gaps

### 3. Urgency-Aware Safety-Critical Reward
- Reward incorporates:
  - Time-to-collision  
  - Lane stability  
  - Velocity smoothness  
  - Traffic density  
- Dynamically prioritizes safety under congestion

### 4. Performance (Dense Traffic Scenario)

| Model        | Cumulative Reward |
|---------------|------------------|
| DQN           | 48.52            |
| Double DQN    | 67.49            |
| D3QN          | 65.74            |
| QCOMBO        | 75.52            |
| **MQLC ** | **120.41**       |

- **2.4× improvement** over standard DQN  
- Higher lane-change success rate  
- Lower collision frequency  
- Longer episode survival in congested traffic  

---

## Environment

Simulated using a customized **highway-env** setup with:

- Multi-lane highway (3–5 lanes)
- Mixed AV–HDV traffic
- Dense scenario: **30 vehicles, 5 controlled AV agents**
- Collision detection, headway penalties, and comfort constraints
- Episode termination based on collision or traffic deadlock

---

## Model Architecture

- **GCN Layer:** Learns spatial dependencies between nearby vehicles  
- **GRU Module:** Models temporal intent evolution  
- **Individual Q-Network:** Outputs local action values  
- **Global Q-Network:** Evaluates joint AV actions  
- **CTDE Training:** Centralized critic, decentralized execution policies  

---

## Training Setup

- Framework: PyTorch + Stable-Baselines3
- Paradigm: CTDE (Centralized Training, Decentralized Execution)
- Discount factor: γ = 0.99  
- Replay buffer with joint-action sampling  
- Reward shaping: safety, efficiency, comfort, urgency  
- Evaluation across multiple random seeds and traffic densities

---

## Applications

- Cooperative Autonomous Driving  
- Lane-Change & Merge Decision-Making  
- Safety-Critical Multi-Agent Control  
- Traffic Simulation and Digital Twins  
- Autonomous Highway Planning Systems  

---

## Installation

```bash
git clone https://github.com/muhammed-arshad-pk/MQLC-Traffic-MARL.git
cd MQLC-Traffic-MARL
pip install -r requirements.txt
