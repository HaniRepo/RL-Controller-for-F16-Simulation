# GenerativeRLController

This repository contains the proof-of-concept implementation for the IEEE IES Generative AI Challenge 2026 Milestone 2 submission:

**Generative Safety Augmentation for Industrial Reinforcement Learning Control Using Conformal Temporal Logic**

The project extends the AeroBench F-16 benchmark with:

* a PPO-based reinforcement learning controller,
* STL-inspired runtime safety monitoring,
* conformal prediction for uncertainty-aware safety assessment,
* a lightweight generative selector for candidate action refinement.

---

## Repository Structure

```text
code/
├── aerobench/                     # Original AeroBench simulator
├── f16_engine_env.py              # F-16 engine Gymnasium environment
├── train_Newppo.py                # PPO training script
├── run_mini_suite.py              # PPO / STL / conformal benchmark suite
├── stress_wrappers.py             # Noise, delay, rate limit, setpoint jump wrappers
├── shield.py                      # Simple STL-style safety shield
├── conformal_shield.py            # Conformal STL runtime shield
├── stl_monitor.py                 # STL robustness utilities
├── shield_pack/
│   └── ppo_f16_engine_baseline.zip # Pretrained PPO baseline
└── genai_hackathon/
    ├── run_genai_suite.py         # Lightweight GenAI proof-of-concept test
    ├── selector_ablation.py       # Candidate selector ablation study
    ├── plot_full_rollout_genai.py # Full rollout figure generation
    └── figures/                   # Generated figures
```

---

## Environment Setup

Recommended: Python 3.10–3.11 in a clean virtual environment.

### 1. Create environment

```bash
python -m venv .venv
```

Activate:

* Windows:

```bash
.venv\Scripts\activate
```

* Linux / macOS:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install numpy scipy matplotlib
pip install gymnasium stable-baselines3 torch
```

Optional (only for AeroBench extras):

```bash
pip install control slycot
```

---

## Quick Start (Recommended for Reviewers)

All commands should be run from inside the `code/` directory.

```bash
cd code
```

### A. Reproduce baseline benchmark suite

This compares:

* PPO baseline
* PPO + STL shield
* PPO + conformal shield

```bash
python run_mini_suite.py
```

Outputs:

* `mini_suite_out/mini_suite.csv`
* `mini_suite_out/mini_suite.json`

---

### B. Run GenAI proof-of-concept selector

This runs the proposed generative action refinement layer.

```bash
python genai_hackathon/run_genai_suite.py
```

Outputs:

* STL satisfaction rate
* mean robustness

---

### C. Generate full rollout figure used in the paper

```bash
python genai_hackathon/plot_full_rollout_genai.py
```

Outputs:

* `genai_hackathon/figures/full_rollout_genai.png`

This figure illustrates:

* airspeed tracking,
* setpoint change adaptation,
* STL tolerance band,
* GenAI runtime interventions.

---

### D. Run selector ablation study

```bash
python genai_hackathon/selector_ablation.py
```

Outputs:

* selector change frequency,
* predicted robustness gains,
* ablation CSV summaries.

---

## Notes for Reviewers

* A pretrained PPO controller is already included (`shield_pack/ppo_f16_engine_baseline.zip`), so no training is required.
* The current implementation is a lightweight proof-of-concept designed for Milestone 2 validation.
* The GenAI module currently uses local candidate action generation with short-horizon conformal robustness evaluation.
* Future work will extend this to richer trajectory synthesis models.

---

## Paper and Submission

This repository accompanies the IEEE IES Generative AI Challenge 2026 submission.

If you are reviewing the manuscript, the repository provides:

* the F-16 benchmark setup,
* the PPO baseline,
* runtime conformal safety shield,
* the proposed GenAI selector proof-of-concept.

GitHub repository:

[https://github.com/HaniRepo/GenerativeRLController](https://github.com/HaniRepo/GenerativeRLController)
