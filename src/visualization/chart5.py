import pandas as pd
import matplotlib.pyplot as plt
import ast
import os
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

os.makedirs("charts", exist_ok=True)

# Load data
df = pd.read_csv("results/qpp_genre_results_4.csv")

# Parse labels
baseline_labels = [ast.literal_eval(l) for l in df["baseline_labels"]]
trs_labels = [ast.literal_eval(l) for l in df["trs_labels"]]

# Take top 10
baseline_top10 = [l[:10] for l in baseline_labels]
trs_top10 = [l[:10] for l in trs_labels]

# Shorten query names
queries = [q[:30] + "..." if len(q) > 30 else q for q in df["query"]]

# --- Light pastel colours ---
colors = ["#ffcccc", "#a6dba6"]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=2)

# --- Create figure with 2 rows (heatmaps on top, colour bar at bottom) ---
fig = plt.figure(figsize=(15, 8))

# Grid: 2 rows (10:1 ratio), 2 columns (equal width)
gs = fig.add_gridspec(
    2, 2, height_ratios=[10, 1], width_ratios=[4.5, 4.5], wspace=0.1, hspace=0.05
)

ax1 = fig.add_subplot(gs[0, 0])  # Baseline
ax2 = fig.add_subplot(gs[0, 1])  # TRS
ax_cbar = fig.add_subplot(gs[1, :])  # Colour bar (spans both columns at the bottom)

# --- MANUAL MARGINS TO FIX LEFT CUT-OFF ---
fig.subplots_adjust(left=0.25, right=0.92, top=0.88, bottom=0.15)

# --- Baseline Heatmap (left) ---
sns.heatmap(
    baseline_top10,
    ax=ax1,
    cmap=cmap,
    vmin=0,
    vmax=1,
    annot=True,
    fmt="d",
    cbar=False,
    square=True,
    linewidths=0.5,
    linecolor="gray",
    xticklabels=[f"#{i + 1}" for i in range(10)],
    yticklabels=queries,
)
ax1.set_title("Baseline: Relevance Labels (Top 10)", fontsize=13)
ax1.set_xlabel("Rank Position")
ax1.tick_params(axis="y", labelsize=9)

# --- TRS Heatmap (right) ---
sns.heatmap(
    trs_top10,
    ax=ax2,
    cmap=cmap,
    vmin=0,
    vmax=1,
    annot=True,
    fmt="d",
    cbar=False,
    square=True,
    linewidths=0.5,
    linecolor="gray",
    xticklabels=[f"#{i + 1}" for i in range(10)],
    yticklabels=False,
)  # REMOVE DUPLICATE QUERY NAMES
ax2.set_title("TRS: Relevance Labels (Top 10)", fontsize=13)
ax2.set_xlabel("Rank Position")
ax2.tick_params(axis="y", left=False, labelleft=False)  # Hide left ticks

# --- HORIZONTAL COLOUR BAR AT THE BOTTOM ---
norm = plt.Normalize(0, 1)
cb = fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    cax=ax_cbar,
    orientation="horizontal",
    label="Relevance (0 = Irrelevant, 1 = Relevant)",
)
cb.ax.tick_params(labelsize=10)
cb.set_ticks([0, 1])
cb.set_ticklabels(["0 (Irrelevant)", "1 (Relevant)"])

# Remove the white border around the colour bar axis
ax_cbar.set_frame_on(False)
ax_cbar.tick_params(axis="both", which="both", length=0)

# Save the image (no tight_layout, because we used manual margins)
plt.savefig("charts/chart5_relevance_heatmap.png", dpi=300)
print("✅ Chart 5 saved to charts/chart5_relevance_heatmap.png")
