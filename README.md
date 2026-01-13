# MQLC: Collaborative Multi-Agent Q-Learning for Autonomous Lane-Changing

## Overview

This repository implements **MQLC (Multi-Agent Q-Learning with Collaboration)**, a research-oriented Multi-Agent Reinforcement Learning (MARL) framework for safe and efficient autonomous lane-changing in mixed traffic environments containing Autonomous Vehicles (AVs) and Human-Driven Vehicles (HDVs).

The framework follows the **Centralized Training with Decentralized Execution (CTDE)** paradigm and combines:

- Individual decision-making using Deep Q-Networks  
- Global coordination via a centralized joint-action Q-Network  
- Graph Convolutional Networks (GCN) for inter-vehicle interaction modeling  
- Urgency-aware reward shaping for collision reduction and traffic efficiency  

The system is evaluated under **sparse, normal, and dense traffic conditions**.

---

## Key Features

- **Hybrid Dual Q-Network Architecture**
  - Individual Q-Network for per-AV actions
  - Global Q-Network for joint coordination
  - Regularization for consistency between local and global value functions

- **Graph-Based Interaction Modeling**
  - Vehicles represented as nodes
  - Edges based on spatial proximity
  - GCN captures collective traffic dynamics

- **Urgency-Aware Reward Function**
  - Considers traffic density, mean speed, and velocity variance
  - Prioritizes safety in congested conditions

- **Multi-Condition Evaluation**
  - Sparse traffic
  - Normal traffic
  - Dense traffic

---

## Environment

A custom multi-lane highway simulator is implemented with:

- Mixed AV–HDV traffic
- Lane-changing and speed control
- Collision detection and proximity penalties
- Adaptive termination based on congestion level

---

## Model Architecture

- **GCN Layer:** Models spatial dependencies between nearby vehicles
- **Individual Q-Network:** Outputs per-agent action values
- **Global Q-Network:** Outputs joint-action value over all AVs
- **CTDE Training:** Centralized learning, decentralized execution

---

## Installation

```bash
git clone https://github.com/muhammed-arshad-pk/MQLC-Traffic-MARL.git
cd MQLC-Traffic-MARL
pip install -r requirements.txt
