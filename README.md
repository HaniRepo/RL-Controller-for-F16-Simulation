<p align="center">
  <img src="anim3d.gif"/>
</p>

# RL Controller for F-16 Simulation

This repository contains a Reinforcement Learning (RL)–based control framework for the F-16 aircraft simulation, built on top of the AeroBench environment. The project extends the original benchmark with learning-based controllers, training pipelines, and evaluation utilities for research and experimentation.

The goal is to study reinforcement learning methods for controlling nonlinear, high-dimensional aerospace systems and to support reproducible benchmarking and analysis.

## Background and Motivation

Aircraft dynamics are nonlinear, high-dimensional (typically 10–20 continuous state variables), and hybrid, with mode-dependent dynamics and discontinuous ordinary differential equations without state jumps. While classical control techniques are well established, reinforcement learning offers a flexible alternative for handling complex objectives, constraints, and uncertainties. This project explores RL-based control strategies applied to the F-16 benchmark model.

## Repository Structure

- code/env – Environment definitions and F-16 simulation wrappers  
- code/controllers – Reinforcement learning controllers and baseline control strategies  
- code/models – Neural network architectures and policy definitions  
- code/training – Training scripts, callbacks, and learning utilities  
- code/evaluation – Evaluation scripts and performance analysis tools  
- code/utils – Helper functions for logging, plotting, and data handling  
- code/configs – Configuration files and hyperparameter settings  
- code/main.py – Main entry point for running training or evaluation  

## Running the Code

To run the simulation and controller, navigate to the code directory and execute:

python main.py

The execution will initialize the F-16 simulation environment, load the selected controller (reinforcement learning or baseline), perform training or evaluation episodes, and log results with optional plots or animations.

## Reinforcement Learning Framework

The reinforcement learning controller interacts with the F-16 simulator through a custom environment interface.

State: aircraft state variables such as angles, angular rates, and velocities.  
Action: control surface commands applied to the aircraft.  
Reward: designed to track reference trajectories while penalizing instability and excessive control effort.

The framework supports training policies from scratch, resuming training from saved checkpoints, and evaluating trained policies on predefined maneuver scenarios.

## Evaluation and Visualization

Evaluation utilities enable systematic testing of trained controllers, logging of aircraft trajectories and control inputs, and visualization of learning behavior and performance metrics. Typical outputs include state and control trajectories, cumulative reward curves, tracking and stability indicators, and optional 2D or 3D animations of aircraft motion.

## Required Libraries

The following Python libraries are required and can be installed using pip:

numpy – matrix operations and numerical computations  
scipy – numerical integration and optimization  
matplotlib – plotting and visualization  
gym or gymnasium – reinforcement learning environment interface  
torch – neural network models and learning algorithms  

Additional libraries may be required depending on the selected controller or visualization setup.

## Reproducibility

Experiments are configuration-driven, random seeds can be fixed, and trained models and logs can be saved and reloaded. For published results, it is recommended to tag the exact code version used in experiments.

## Citation

If you use this code in academic work, please cite:

H. Beirami,  M. Manjurul Islam,
Conformal Signal Temporal Logic for Robust Reinforcement Learning Control: A Case Study, to be appeared in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026.

## Citation to the case study
"Verification Challenges in F-16 Ground Collision Avoidance and Other Automated Maneuvers", P. Heidlauf, A. Collins, M. Bolender, S. Bak, 5th International Workshop on Applied Verification for Continuous and Hybrid Systems (ARCH 2018)

## License

This project is released under the terms of the license provided in the LICENSE file.
