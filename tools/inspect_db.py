import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import pandas as pd
from tabulate import tabulate

from src.ingestion.static_loader import get_vectorstore


def main():
    parser = argparse.ArgumentParser(description="Inspect ChromaDB srored data")
    parser.add_argument("--csv", type=str, help="Save ouutput to a CSV file")
    args = parser.parse_args()

    # connect to the exissting database
    vectorstore = get_vectorstore()
    data = vectorstore.get(include=["metadatas", "documents"])

    # build datframe
    df = pd.DataFrame(data["metadatas"])
    df["content"] = data["documents"]

    # select and order columns
    columns = [
        "id",
        "entity_type",
        "source_type",
        "credibility",
        "document_type",
        "timestamp",
        "location",
        "content",
    ]

    df = df[columns]

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"✅ Data exported ro {args.csv} ({len(df)} rows)")
    else:
        print(tabulate(df, headers="keys", tablefmt="simple", showindex=False))


if __name__ == "__main__":
    main()
