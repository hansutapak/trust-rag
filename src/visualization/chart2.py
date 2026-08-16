import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.makedirs("charts", exist_ok=True)

# Load your results
df = pd.read_csv("results/qpp_genre_results_4.csv")

# Shorten query names
queries = []
for q in df["query"]:
    if len(q) > 40:
        queries.append(q[:40] + "...")
    else:
        queries.append(q)

y_pos = np.arange(len(queries))
jitter = 0.02

fig, ax = plt.subplots(figsize=(10, 6))

# --- 1. DRAW THE DASHED LINES FIRST (so they go behind the markers) ---
for i in range(len(queries)):
    ax.plot(
        [df["baseline_ndcg10"][i], df["trs_ndcg10"][i]],
        [i, i],
        color="gray",
        linestyle="--",
        alpha=0.5,
        linewidth=1,
    )

# --- 2. DRAW THE MARKERS ON TOP ---
# Baseline (pink circles, shifted left)
ax.scatter(
    df["baseline_ndcg10"] - jitter,
    y_pos,
    label="Baseline",
    color="#ff9999",
    s=120,
    marker="o",
    edgecolor="black",
    linewidth=0.5,
    zorder=5,
)

# TRS (blue diamonds, shifted right)
ax.scatter(
    df["trs_ndcg10"] + jitter,
    y_pos,
    label="TRS",
    color="#66b3ff",
    s=120,
    marker="D",
    edgecolor="black",
    linewidth=0.5,
    zorder=5,
)

# --- 3. GRID AND LABELS ---
ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.8)

ax.set_yticks(y_pos)
ax.set_yticklabels(queries, fontsize=9)
ax.set_xlabel("nDCG@10 Score", fontsize=11)
ax.set_title("Per-Query Comparison: Baseline vs. TRS", fontsize=13)

# --- 4. MOVE LEGEND TO TOP LEFT (away from the high scores) ---
ax.legend(loc="upper left")

ax.set_xlim(-0.05, 1.05)

plt.tight_layout()
plt.savefig("charts/chart2_per_query.png", dpi=300)
print("✅ Chart 2 updated: Legend moved to top left, markers are on top of lines.")
