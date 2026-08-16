def normalize_raw_reading(raw: dict) -> dict:
    """
    convert a raw sensor reading into standard NGSI-LD format
    """

    return {
        "id": raw["sensor_id"],
        "type": "Sensor",
        "location": {
            "type": "Point",
            "coordinates": [raw["lon"], raw["lat"]],
        },
        "dateObserved": raw["timestamp"],
        "source": {
            "dataset": "live_sensors",
            "sensorType": raw["source_type"],
        },
        "attributes": {
            "metric": raw["metric"],
            "reading": raw["reading"],
        },
    }
