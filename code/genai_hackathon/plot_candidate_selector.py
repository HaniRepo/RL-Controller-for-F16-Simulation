import os
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

# your actual snapshot values
actions = np.array([0.113, 0.143, 0.173, 0.213])
robustness = np.array([0.032557, 0.032604, 0.032651, 0.032714])

best_idx = np.argmax(robustness)

plt.figure(figsize=(6.5, 4.2))

bars = plt.bar(actions, robustness, width=0.018)

# highlight best candidate
bars[best_idx].set_linewidth(2)

# PPO baseline marker
plt.axvline(actions[0], linestyle="--", linewidth=2, label="PPO baseline action")

# best selected marker
plt.scatter(actions[best_idx], robustness[best_idx], s=120, zorder=5, label="GenAI selected")

for x, y in zip(actions, robustness):
    plt.text(x, y + 0.00001, f"{y:.4f}", ha="center", fontsize=9)

plt.xlabel("Candidate throttle action")
plt.ylabel("Predicted worst-case STL robustness")
plt.title("GenAI runtime action selection")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(OUTDIR, "genai_candidate_selector.png"), dpi=220)
plt.show()