<p align="center">
  <img src="anim3d.gif"/>
</p>

# RL Controller for F-16 Simulation

This repository contains a Reinforcement Learning (RL)–based control framework for the F-16 aircraft simulation, built on top of the AeroBench environment. The project extends the original benchmark with learning-based controllers, training pipelines, safety shielding, and evaluation utilities for research and experimentation.

The goal is to study reinforcement learning methods for controlling nonlinear, high-dimensional aerospace systems and to support reproducible benchmarking, robustness analysis, and safety-aware evaluation.

---

## Background and Motivation

Aircraft dynamics are nonlinear, high-dimensional (typically 10–20 continuous state variables), and hybrid, with mode-dependent dynamics and discontinuous ordinary differential equations without state jumps. While classical control techniques are well established, reinforcement learning offers a flexible alternative for handling complex objectives, constraints, and uncertainties.

This project explores RL-based control strategies applied to the F-16 benchmark model, with additional emphasis on robustness, stress testing, and formal/safety-inspired monitoring.

---

## Repository Structure

All source code is located under the `code/` directory.

### Core AeroBench Simulator (Upstream)

- **aerobench/lowlevel/**  
  Low-level aircraft and engine models, aerodynamic coefficient tables, and nonlinear dynamics.

- **aerobench/highlevel/**  
  Higher-level autopilot and controller interfaces.

- **aerobench/visualize/**  
  Plotting and animation utilities (2D and 3D).

- **aerobench/examples/**  
  Ready-to-run benchmark scenarios (GCAS, waypoint tracking, straight-and-level, ACAS Xu, animations).

### RL + Safety / Benchmark Layer (This Repository)

- **f16_engine_env.py**  
  Gymnasium-compatible environment for F-16 engine control and airspeed tracking.

- **train_Newppo.py**  
  Main training script: trains a PPO controller (Stable-Baselines3), saves models, generates plots, and evaluates safety- and conformal-shielded rollouts.

- **reproduce_baseline.py**  
  Loads a pretrained PPO baseline from `baseline_pack/` and reproduces reference results.

- **run_mini_suite.py**  
  Lightweight benchmark suite for quick comparison between PPO and shielded variants.

- **run_benchmark_suite.py**  
  Full benchmark suite with stress testing (noise, delay, setpoint jumps, throttle caps, rate limits) and CSV outputs.

- **stress_wrappers.py**  
  Environment wrappers that inject stress and fault conditions.

- **shield.py**  
  Lightweight safety filter applied on top of a base policy.

- **conformal_shield.py**  
  Conformal prediction–based shield enforcing STL-style constraints under uncertainty.

- **stl_monitor.py**  
  Signal Temporal Logic (STL) monitoring utilities (e.g., settling specifications).

- **utils_plot.py, make_figures.py, plot_stress_compare.py, eval_seeds.py**  
  Utilities for evaluation, plotting, and paper-quality figure generation.

---

## Running the Code

All runnable scripts are located inside the `code/` directory.

**Important:** Always run scripts from inside `code/` so that imports such as `from aerobench...` work correctly.

```bash
cd code
```

### 1) Install Dependencies

It is recommended to use a virtual environment.
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```
Install required packages:
```bash
pip install numpy scipy matplotlib
pip install gymnasium stable-baselines3 torch
```
Optional (not required for RL training, only for some AeroBench control-design utilities):
```bash
pip install control slycot
```

### 2) Train the RL Controller (Main Entry Point)
You can use the pretrained model or train the model from the scratch(about 2-10 minutes)
This script trains a PPO controller on the F-16 engine environment and evaluates it with safety and conformal shielding mechanisms.
```bash
python train_Newppo.py
```
Optional arguments:
```bash
python train_Newppo.py \
  --setpoint 500 \
  --dt 0.1 \
  --horizon 60 \
  --steps 300000 \
  --model-path ./ppo_f16_engine.zip
```
Typical outputs include:

- trained PPO model (.zip)
- tracking plots (e.g., vt_tracking.png)
- shielded and conformal evaluation plots

### 3) Reproduce the Baseline Controller
This script loads a pretrained PPO baseline from baseline_pack/ and reproduces a reference run.
```bash
python reproduce_baseline.py
```
Outputs include:
- reproduced tracking plots
- printed STL satisfaction results

### 4) Run Benchmark Suites
Mini Benchmark Suite
```bash
python run_mini_suite.py
```
Full Benchmark Suite
```bash
python run_benchmark_suite.py
```
These benchmarks compare:
- PPO baseline
- PPO + STL shield
- PPO + conformal/STL shield

under stress conditions such as noise, delay, setpoint jumps, throttle limits, and action-rate constraints.
Results are saved to output folders and CSV files for analysis.

### 5) (Optional) Run Original AeroBench Examples
The original AeroBench maneuver scripts remain available.

Examples:
```bash
python aerobench/examples/gcas/run_GCAS.py
python aerobench/examples/waypoint/run_waypoint.py
python aerobench/examples/anim3d/run_GCAS_anim3d.py
```


## Reinforcement Learning Framework
The reinforcement learning controller interacts with the F-16 simulator through a custom environment interface.

- State: aircraft state variables such as angles, angular rates, and velocities
- Action: control surface commands applied to the aircraft
- Reward: designed to track reference trajectories while penalizing instability and excessive control effort

The framework supports training policies from scratch, resuming training from saved checkpoints, and evaluating trained policies on predefined maneuver scenarios.

## Evaluation and Visualization
Evaluation utilities enable systematic testing of trained controllers, logging of aircraft trajectories and control inputs, and visualization of learning behavior and performance metrics. Typical outputs include state and control trajectories, cumulative reward curves, tracking and stability indicators, and optional 2D or 3D animations of aircraft motion.
### Required Libraries
The following Python libraries are required and can be installed using pip:

- numpy – matrix operations and numerical computations
- scipy – numerical integration and optimization
- matplotlib – plotting and visualization
- gym or gymnasium – reinforcement learning environment interface
- torch – neural network models and learning algorithms
Additional libraries may be required depending on the selected controller or visualization setup.

## Reproducibility

Experiments are configuration-driven, random seeds can be fixed, and trained models and logs can be saved and reloaded. For published results, it is recommended to tag the exact code version used in experiments.

## Citation

If you use this code in academic work, please cite:

H. Beirami, M. M. Manjurul Islam,
Conformal Signal Temporal Logic for Robust Reinforcement Learning Control: A Case Study,
IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2026.

### Citation to the Case Study
"Verification Challenges in F-16 Ground Collision Avoidance and Other Automated Maneuvers",
P. Heidlauf, A. Collins, M. Bolender, S. Bak,
5th International Workshop on Applied Verification for Continuous and Hybrid Systems (ARCH 2018).
