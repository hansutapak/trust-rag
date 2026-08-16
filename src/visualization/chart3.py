import os

import matplotlib.pyplot as plt

os.makedirs("charts", exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 2.5))
ax.axis("tight")
ax.axis("off")

# Table data – taken directly from your debug output
data = [
    [
        "Which AQI to trust?",
        "BCN-SM-001 (Citizen)",
        "BCN-AQ-001 (Government)",
        "✅ TRS picked correct source",
    ],
    [
        "Conflict AQI-999",
        "BCN-AQ-999 (Faulty)",
        "BCN-AQ-001 (Good)",
        "✅ TRS filtered outlier",
    ],
]
columns = ["Query", "Baseline Top ID", "TRS Top ID", "Verdict"]

# Create the table
table = ax.table(cellText=data, colLabels=columns, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

# Highlight the "Verdict" column in green
for i, row in enumerate(data, 1):
    if "✅" in row[3]:
        table[(i, 3)].set_facecolor("#d4edda")  # light green

# Title
ax.set_title(
    "Sanity Check: TRS Retrieval vs. LLM Strictness\n(Despite LLM zeros, TRS still retrieved the exact correct document)",
    fontsize=11,
    pad=20,
)

plt.tight_layout()
plt.savefig("charts/chart3_sanity_table.png", dpi=300)
print("✅ Chart 3 (Sanity Check Table) saved to charts/chart3_sanity_table.png")
