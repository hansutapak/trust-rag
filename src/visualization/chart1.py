import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.makedirs("charts", exist_ok=True)

# Load your results
df = pd.read_csv("results/qpp_genre_results_4.csv")

# Calculate means
metrics = ["RR@10", "nDCG@10"]
baseline = [df["baseline_rr10"].mean(), df["baseline_ndcg10"].mean()]
trs = [df["trs_rr10"].mean(), df["trs_ndcg10"].mean()]

# Create bar chart
fig, ax = plt.subplots(figsize=(8, 6))
x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width / 2, baseline, width, label="Baseline", color="#ff9999")
bars2 = ax.bar(x + width / 2, trs, width, label="TRS", color="#66b3ff")

# Add value labels
for bar in bars1:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{bar.get_height():.3f}",
        ha="center",
        va="bottom",
    )
for bar in bars2:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{bar.get_height():.3f}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

# Add improvement annotations
for i, m in enumerate(metrics):
    improvement = (trs[i] - baseline[i]) / baseline[i] * 100
    ax.text(
        i,
        max(baseline[i], trs[i]) + 0.04,
        f"+{improvement:.0f}%",
        ha="center",
        fontweight="bold",
        color="darkgreen",
        fontsize=12,
    )

ax.set_ylabel("Score")
ax.set_title("Baseline vs. TRS (Mean over 8 Queries)")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.7)
ax.set_ylim(0, 0.5)

plt.tight_layout()
plt.savefig("charts/chart1_improvement.png", dpi=300)
print("✅ Chart 1 saved to charts/chart1_improvement.png")
