import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Ensure charts directory exists
os.makedirs("charts", exist_ok=True)

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
df = pd.read_csv("results/answer_level_manual_final.csv")


# ----------------------------------------------------------------------
# 2. Add a "category" column for grouping
# ----------------------------------------------------------------------
def get_category(query):
    q = query.lower()
    if "aqi" in q:
        return "AQI"
    elif "temperature" in q or "temp" in q:
        return "Temperature"
    elif "traffic" in q:
        return "Traffic"
    elif "bin" in q:
        return "Bin Fill"
    elif "tree" in q or "species" in q or "pinus" in q:
        return "Static (Trees)"
    elif "address" in q or "green point" in q:
        return "Static (Addresses)"
    else:
        return "Conflict / Mixed"


df["category"] = df["query"].apply(get_category)

# ----------------------------------------------------------------------
# 3. Chart 1: Headline Bar Chart
# ----------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(6, 5))
bars = ax1.bar(
    ["Baseline", "TRS"],
    [df["baseline_partial"].mean(), df["trs_partial"].mean()],
    color=["#ff9999", "#66b3ff"],
    edgecolor="black",
    linewidth=1,
)

for bar in bars:
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.01,
        f"{height:.1%}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

ax1.set_ylabel("Partial Match Rate")
ax1.set_title("Answer‑Level Partial Match: Baseline vs TRS")
ax1.set_ylim(0, 1)
ax1.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("charts/answer_level_chart1.png", dpi=300)
print("✅ Chart 1: Main bar chart saved.")

# ----------------------------------------------------------------------
# 4. Chart 2: Per-Query Dot Plot
# ----------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(12, 8))  # wider for readability
queries_short = df["query"].apply(lambda x: x[:30] + "..." if len(x) > 30 else x)
y_pos = range(len(queries_short))

jitter = 0.02  # small horizontal offset to separate overlapping points

# Draw connecting lines FIRST (zorder=1) so they go behind the dots
for i in range(len(df)):
    ax2.plot(
        [df.iloc[i]["baseline_partial"], df.iloc[i]["trs_partial"]],
        [i, i],
        color="gray",
        linestyle="--",
        alpha=0.5,
        linewidth=1,
        zorder=1,
    )

# Draw dots SECOND (zorder=3) on top of lines
# Baseline shifted LEFT by jitter
ax2.scatter(
    df["baseline_partial"] - jitter,
    y_pos,
    label="Baseline",
    color="#ff9999",
    s=120,
    edgecolor="black",
    linewidth=0.5,
    zorder=3,
)
# TRS shifted RIGHT by jitter
ax2.scatter(
    df["trs_partial"] + jitter,
    y_pos,
    label="TRS",
    color="#66b3ff",
    s=120,
    marker="D",
    edgecolor="black",
    linewidth=0.5,
    zorder=3,
)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(queries_short, fontsize=7)
ax2.set_xlabel("Partial Match (0 = No, 1 = Yes)")
ax2.set_title("Per‑Query Partial Match: Baseline vs TRS")
ax2.legend()
ax2.grid(axis="x", linestyle="--", alpha=0.3)
ax2.set_xlim(-0.1, 1.1)  # ensure dots at edges are visible
plt.tight_layout()
plt.savefig("charts/answer_level_chart2.png", dpi=300, bbox_inches="tight")
print("✅ Chart 2: Per-query plot saved (jitter added, dots on top).")

# ----------------------------------------------------------------------
# 5. Chart 3: Improvement Delta (TRS - Baseline) per query
# ----------------------------------------------------------------------
df["delta"] = df["trs_partial"] - df["baseline_partial"]
df_sorted = df.sort_values("delta", ascending=True)

fig3, ax3 = plt.subplots(figsize=(10, 8))
colors = ["green" if x > 0 else "red" if x < 0 else "gray" for x in df_sorted["delta"]]
bars = ax3.barh(
    df_sorted["query"].apply(lambda x: x[:30] + "..."), df_sorted["delta"], color=colors
)

ax3.axvline(x=0, color="black", linestyle="-", alpha=0.5)
ax3.set_xlabel("Delta (TRS - Baseline)")
ax3.set_title("Query Improvement: Which Queries Did TRS Help Most?")
ax3.grid(axis="x", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("charts/answer_level_chart3.png", dpi=300)
print("✅ Chart 3: Improvement delta saved.")

# ----------------------------------------------------------------------
# 6. Chart 4: Category Performance (Grouped Bar Chart)
# ----------------------------------------------------------------------
cat_means = (
    df.groupby("category")[["baseline_partial", "trs_partial"]].mean().reset_index()
)
fig4, ax4 = plt.subplots(figsize=(10, 6))
x = np.arange(len(cat_means))
width = 0.35

ax4.bar(
    x - width / 2,
    cat_means["baseline_partial"],
    width,
    label="Baseline",
    color="#ff9999",
)
ax4.bar(x + width / 2, cat_means["trs_partial"], width, label="TRS", color="#66b3ff")

ax4.set_xticks(x)
ax4.set_xticklabels(cat_means["category"])
ax4.set_ylabel("Partial Match Rate")
ax4.set_title("Category Performance: Baseline vs TRS")
ax4.legend()
ax4.grid(axis="y", linestyle="--", alpha=0.3)
ax4.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig("charts/answer_level_chart4.png", dpi=300)
print("✅ Chart 4: Category performance saved.")

# ----------------------------------------------------------------------
# 7. Chart 5: Heatmap (Baseline vs TRS side by side)
# ----------------------------------------------------------------------
# Only take top 15 queries for readability
heatmap_df = df.head(15)
fig5, ax5 = plt.subplots(figsize=(8, 6))
heatmap_data = heatmap_df[["baseline_partial", "trs_partial"]].T
sns.heatmap(
    heatmap_data,
    cmap="RdYlGn",
    vmin=0,
    vmax=1,
    annot=True,
    fmt="d",
    cbar_kws={"label": "Partial Match (0/1)"},
    xticklabels=heatmap_df["query"].apply(lambda x: x[:20] + "..."),
    yticklabels=["Baseline", "TRS"],
)
ax5.set_title("Heatmap: Baseline vs TRS Partial Matches (Top 15)")
plt.tight_layout()
plt.savefig("charts/answer_level_chart5.png", dpi=300)
print("✅ Chart 5: Heatmap saved.")

print("\n🎉 All 5 charts saved to the 'charts/' folder.")
