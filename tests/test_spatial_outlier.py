"""Inject an extreme outlier to verify penalty < 1 and Bayesian credibility drops."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document

from src.ingestion.static_loader import get_vectorstore
from src.scoring.trs import apply_spatial_conflict_penalty, get_bayesian_credibility

OUTLIER_ID = "BCN-AQ-999"
OUTLIER_META = {
    "timestamp": "2026-07-05T16:15:00+00:00",
    "source_type": "government_sensor",
    "credibility": 0.9,
    "document_type": "sensor",
    "entity_type": "sensor",
    "location": [2.165, 41.389],
    "id": OUTLIER_ID,
}


def main():
    vs = get_vectorstore()

    # 1. Inject the outlier if not already present
    existing = vs.similarity_search(OUTLIER_ID, k=1)
    if not existing or existing[0].metadata.get("id") != OUTLIER_ID:
        doc = Document(
            page_content=f"sensor: {OUTLIER_ID} | AQI: 999 | Location: 41.389, 2.165",
            metadata=OUTLIER_META,
        )
        vs.add_documents([doc])
        print("Injected outlier chunk.")
    else:
        print("Outlier already present.")

    # 2. Find it and compute penalty
    docs = vs.similarity_search(OUTLIER_ID, k=1)
    chunk = docs[0]
    penalty = apply_spatial_conflict_penalty(chunk, vs)
    print(f"Penalty for {OUTLIER_ID}: {penalty:.4f}  (expected < 1.0)")

    # 3. Check Bayesian credibility
    cred_outlier = get_bayesian_credibility("government_sensor", OUTLIER_ID)
    cred_normal = get_bayesian_credibility("government_sensor", "BCN-AQ-001")
    print(f"Credibility {OUTLIER_ID}: {cred_outlier:.4f}  (expected < 0.9)")
    print(f"Credibility BCN-AQ-001:  {cred_normal:.4f}  (expected >= 0.9)")

    if penalty < 1.0 and cred_outlier < 0.9:
        print("\nSpatial conflict resolution: WORKING")
    else:
        print("\nUnexpected values – check the function.")


if __name__ == "__main__":
    main()
