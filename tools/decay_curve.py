import math

import matplotlib.pyplot as plt

# λ values and labels
lamdas = [
    (0.0083, "citizen_device (2 min shelf)"),
    (0.0033, "council_approved (5 min)"),
    (0.0011, "commetial_iot (15 min)"),
    (0.0006, "government_sensor (30 min)"),
    (0.0003, "met_office (1 hr)"),
]

# Time range: 0 to 2 hours in steps of 10 seconds
total_seconds = 7200
time_steps = range(0, total_seconds, 10)

plt.figure(figsize=(10, 6))

for lam, label in lamdas:
    recency = [math.exp(-lam * t) for t in time_steps]
    plt.plot([t / 60 for t in time_steps], recency, label=label)

plt.axhline(y=0.37, color="gray", linestyle="--", label="1/e threshold")
plt.xlabel("Age (minutes)")
plt.ylabel("Recency score")
plt.title("Adaptive Temporal Decay Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("charts/adpaptive_decay_curve.png", dpi=200)
plt.grid(True)
plt.show()
