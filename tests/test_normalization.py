"""Test the normaliser"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.etl.normalize import normalize_raw_reading


def _print_example(raw, label):
    """Print a raw message and its normalised NGSI‑LD, then run the output."""
    print(f"=== Raw message ({label}) ===")
    print(raw)
    data = normalize_raw_reading(raw)
    print(f"\n=== Normalised NGSI‑LD ({label}) ===")
    print(data)
    print()

    # --- assertions ---
    assert data["id"] == raw["sensor_id"], "id mismatch"
    assert data["type"] == "Sensor", "type must be 'Sensor'"
    assert data["location"]["type"] == "Point", "location.type must be 'Point'"
    assert data["location"]["coordinates"] == [raw["lon"], raw["lat"]], (
        "coordinates mismatch"
    )
    assert data["dateObserved"] == raw["timestamp"], "timestamp mismatch"
    assert data["source"]["dataset"] == "live_sensors", "dataset wrong"
    assert data["source"]["sensorType"] == raw["source_type"], "source_type mismatch"
    assert data["attributes"]["metric"] == raw["metric"], "metric mismatch"
    assert data["attributes"]["reading"] == raw["reading"], "reading mismatch"

    print(f"✅ {label} normalisation OK\n")


if __name__ == "__main__":
    # 1. Numeric sensor
    raw_sensor = {
        "sensor_id": "BCN-AQ-001",
        "location_name": "Eixample",
        "source_type": "government_sensor",
        "metric": "AQI",
        "reading": 42.7,
        "timestamp": "2026-07-11T10:00:00+00:00",
        "lat": 41.389,
        "lon": 2.165,
    }
    _print_example(raw_sensor, "numeric sensor")

    # 2. Social report
    raw_social = {
        "sensor_id": "BCN-SM-001",
        "location_name": "Eixample",
        "source_type": "citizen_device",
        "metric": "social_report",
        "reading": "Just cycled through Eixample - air feels fresh!",
        "timestamp": "2026-07-11T10:01:00+00:00",
        "lat": 41.389,
        "lon": 2.165,
    }
    _print_example(raw_social, "social report")

    # 3. Raw‑message key check
    required_keys = {
        "sensor_id",
        "location_name",
        "source_type",
        "metric",
        "reading",
        "timestamp",
        "lat",
        "lon",
    }
    raw_check = {
        "sensor_id": "BCN-WE-001",
        "location_name": "Nou Barris",
        "source_type": "met_office",
        "metric": "temperature",
        "reading": 65.5,
        "timestamp": "2026-07-11T10:02:00+00:00",
        "lat": 41.440,
        "lon": 2.181,
    }
    assert set(raw_check.keys()) == required_keys
    print("✅ Raw message keys correct")

    print("\n🎉 All normalisation tests passed.")
