import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.makedirs("charts", exist_ok=True)

df = pd.read_csv("results/qpp_genre_results_4.csv")

# Calculate improvement (delta)
df["delta"] = df["trs_ndcg10"] - df["baseline_ndcg10"]

# Sort by delta (largest improvement at top)
df_sorted = df.sort_values("delta", ascending=True)

# Shorten query names
queries = []
for q in df_sorted["query"]:
    if len(q) > 40:
        queries.append(q[:40] + "...")
    else:
        queries.append(q)

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(queries))

# Colors
colors = [
    "green" if x > 0.01 else "red" if x < -0.01 else "gray" for x in df_sorted["delta"]
]

bars = ax.barh(
    y_pos, df_sorted["delta"], color=colors, edgecolor="black", linewidth=0.5
)

# Add value labels with proper positioning
for i, (bar, delta) in enumerate(zip(bars, df_sorted["delta"])):
    if delta > 0:
        # Positive improvement: label on the right of the bar
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{delta:.3f}",
            va="center",
            ha="left",
            fontweight="bold",
        )
    elif delta < 0:
        # Negative improvement: label on the left of the bar
        ax.text(
            bar.get_width() - 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{delta:.3f}",
            va="center",
            ha="right",
            fontweight="bold",
        )
    else:
        # Zero improvement: label exactly at 0
        ax.text(
            0,
            bar.get_y() + bar.get_height() / 2,
            "0.000",
            va="center",
            ha="center",
            fontweight="bold",
        )

# Vertical line at 0
ax.axvline(x=0, color="black", linestyle="-", alpha=0.5)

# Set x-axis limits with padding to avoid clipping
max_delta = max(abs(df_sorted["delta"]))
padding = 0.1 if max_delta > 0 else 0.1
ax.set_xlim(-max_delta - padding, max_delta + padding)

ax.set_yticks(y_pos)
ax.set_yticklabels(queries, fontsize=9)
ax.set_xlabel("nDCG@10 Improvement (TRS - Baseline)", fontsize=11)
ax.set_title("Per-Query Improvement: Which Queries Did TRS Help Most?", fontsize=13)
ax.grid(axis="x", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("charts/chart4_improvement_delta.png", dpi=300)
print("✅ Chart 4 fixed: labels now inside the axis, no clipping.")
