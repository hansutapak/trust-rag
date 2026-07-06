"""check if spatial conflict resolution runs without errors and returns valid penalties."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.static_loader import get_vectorstore
from src.scoring.trs import apply_spatial_conflict_penalty


def main():
    vs = get_vectorstore()

    # get a fresh chunk that contains numeric metrics
    for metric in ["AQI", "temperature", "traffic_density", "bin_fill"]:
        docs = vs.similarity_search(metric, k=1)
        if not docs:
            print(f"{metric:20s} - no chunk found in the database")
            continue
        chunk = docs[0]
        penalty = apply_spatial_conflict_penalty(chunk, vs)
        print(f"{metric:20s} penalty = {penalty:.4f} (1.0 = no conflict)")

    print(
        "\nIf all penalty values are between 0.1 and 1.0, spatial conflict resolution is operational."
    )

    print("If any value is < 1.0, a spatial outlier was detected and penalised.")


if __name__ == "__main__":
    main()
