import json
import os
import traceback
from datetime import datetime

import pandas as pd

# paths
STATIC_DIR = "data/static"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Possible encodings to try, in order
ENCODINGS = ["utf-8", "utf-16", "utf-8-sig", "latin1"]

# data configurations
DATASETS = {
    "trees": {
        "filename": "opendatabcn_trees.csv",
        "id_field": "codi",
        "type": "Tree",
        "lon_field": "longitud",
        "lat_field": "latitud",
        "date_field": "data_plantacio",
        "fallback_date": None,
        "sensorType": "government",
    },
    "climate": {
        "filename": "opendatabcn_climate_refuge_network.csv",
        "id_field": "register_id",
        "type": "ClimateShelter",
        "lon_field": "geo_epgs_4326_lon",
        "lat_field": "geo_epgs_4326_lat",
        "date_field": "modified",
        "fallback_date": None,
        "sensorType": "mixed",
    },
    "greenpoints": {
        "filename": "opendatabcn_neighbourhood_green_points.csv",
        "id_field": "register_id",
        "type": "GreenPoint",
        "lon_field": "geo_epgs_4326_lon",
        "lat_field": "geo_epgs_4326_lat",
        "date_field": "modified",
        "fallback_date": None,
        "sensorType": "government",
    },
}


def read_csv_with_encoding(filepath):
    """Try to read a CSV using several encodings, return the DataFrame."""
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(filepath, dtype=str, keep_default_na=False, encoding=enc)
            print(f"✅Read with encoding: {enc}")
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Could not read {filepath} with any encoding: {ENCODINGS}")


def build_entity(row, config):
    """Convert a CSV row to an NGSI‑LD‑compatible JSON object."""

    # ID
    entity_id = str(row[config["id_field"]])

    # Location
    lon = float(row[config["lon_field"]])
    lat = float(row[config["lat_field"]])

    # Timestamp
    date_val = row.get(config["date_field"], None)
    if pd.isna(date_val) or date_val in ("", " ", "0000-00-00"):
        if config["fallback_date"] and config["fallback_date"] in row:
            date_val = row[config["fallback_date"]]
        else:
            date_val = datetime.now().isoformat()

    # Build entity
    entity = {
        "id": entity_id,
        "type": config["type"],
        "location": {"type": "Point", "coordinates": [lon, lat]},
        "dateObserved": date_val,
        "source": {
            "dataset": config["type"].lower(),
            "institution": row.get("institution_name", "Unknown"),
            "sensorType": config["sensorType"],
        },
        "attributes": {},
    }

    # Store all other columns as a attributes
    for col in row.index:
        if col not in [
            config["id_field"],
            config["lon_field"],
            config["lat_field"],
            config["date_field"],
            config.get("fallback_date", ""),
            "institution_name",
        ]:
            val = row[col]
            if isinstance(val, (int, float, str, bool, type(None))):
                entity["attributes"][col] = val if not pd.isna(val) else None

            else:
                entity["attributes"][col] = str(val)

    return entity


def convert_dataset(name, config):
    """Read a CSV file, convert all rows, and save as JSON array."""
    input_path = os.path.join(STATIC_DIR, config["filename"])
    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        return

    print(f"🔃 Converting {name}...")
    df = read_csv_with_encoding(input_path)

    entities = []
    for i, row in df.iterrows():
        try:
            entity = build_entity(row, config)
            entities.append(entity)
        except Exception:
            print(f"⚠️ Skipping row {i} due to error:")
            traceback.print_exc()

    # Save to JSON
    output_path = os.path.join(PROCESSED_DIR, f"{name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entities, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(entities)} entities to {output_path}")


if __name__ == "__main__":
    for name, config in DATASETS.items():
        convert_dataset(name, config)
    print("🎉 All datasets converted.")
