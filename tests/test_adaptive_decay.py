"""Check if the adaptive decay applies different λ per source type."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.static_loader import get_vectorstore
from src.scoring.trs import calculate_recency_score, get_adaptive_decay_rate

SOURCE_TYPES = [
    "government_sensor",
    "met_office",
    "council_approved",
    "commercial_iot",
    "citizen_device",
]


def main():
    vs = get_vectorstore()
    print("Adaptive Decay Verification\n")

    for stype in SOURCE_TYPES:
        # Grab one fresh chunk of this type
        docs = vs.similarity_search(stype, k=1)
        if not docs:
            print(f"{stype:20s} – no data found")
            continue
        doc = docs[0]
        ts = doc.metadata["timestamp"]
        decay_rate = get_adaptive_decay_rate(None, "sensor", stype)
        recency = calculate_recency_score(ts, decay_rate)
        print(f"{stype:20s}   λ={decay_rate:.4f}   recency={recency:.4f}")

    print("\n→ If recency values differ, adaptive decay is working.")
    print(
        "→ Higher λ (citizen_device) should give LOWER recency than smaller λ (met_office)."
    )


if __name__ == "__main__":
    main()
