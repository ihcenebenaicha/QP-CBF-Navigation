# Custom CBF-based Robot Navigation

This project implements a **custom Control Barrier Function (CBF) approach** for autonomous robot navigation in environments with obstacles. The algorithm is developed from scratch and provides real-time safe navigation for mobile robots using ROS 2.

---

## Features

- **Dynamic Obstacle Avoidance:** Uses LIDAR data to detect obstacles and generate collision-free velocity commands.
- **Virtual Obstacle Modeling:** Detects dense obstacle clusters and represents them as circular virtual obstacles.
- **Goal-Oriented Navigation:** Moves the robot towards a predefined goal while respecting safety constraints.
- **Quadratic Programming (QP) Control:** Computes linear and angular velocities using CBF-based QP optimization.
- **Data Logging & Visualization:** Stores and plots robot trajectories, velocities, and obstacle positions.
- **ROS 2 Integration:** Compatible with ROS 2 for real-time robot control.

---

## Requirements

- Python
- ROS 2 Humble
- Libraries:
  ```bash
  pip install numpy sympy scipy matplotlib cvxpy
  
