import json
import random
import time
from datetime import UTC, datetime

from faker import Faker
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers="127.0.0.1:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

SENSORS = [
    {
        "sensor_id": "BCN-AQ-001",
        "location": "Eixample",
        "source_type": "government_sensor",
        "metric": "AQI",
        "lat": 41.389,
        "lon": 2.165,
    },
    {
        "sensor_id": "BCN-TR-001",
        "location": "Ciutat Vella",
        "source_type": "council_approved",
        "metric": "traffic_density",
        "lat": 41.380,
        "lon": 2.190,
    },
    {
        "sensor_id": "BCN-WE-001",
        "location": "Nou Barris",
        "source_type": "met_office",
        "metric": "temperature",
        "lat": 41.440,
        "lon": 2.181,
    },
    {
        "sensor_id": "BCN-SM-001",
        "location": "Eixample",
        "source_type": "citizen_device",
        "metric": "social_report",
        "lat": 41.389,
        "lon": 2.165,
    },
    {
        "sensor_id": "BCN-WB-001",
        "location": "Ciutat Vella",
        "source_type": "commercial_iot",
        "metric": "bin_fill",
        "lat": 41.380,
        "lon": 2.190,
    },
]

TOPIC = "smart_city_sensors"


def simulate():
    while True:
        for sensor in SENSORS:
            # generate reading
            if sensor["metric"] == "social_report":
                fake = Faker()
                reading = f"Ctizen report from {sensor['location']}: {fake.sentence()}"
            else:
                reading = round(random.uniform(30, 80), 2)

            # NGSI-LD format
            message = {
                "sensor_id": sensor["sensor_id"],
                "location_name": sensor["location"],
                "source_type": sensor["source_type"],
                "metric": sensor["metric"],
                "reading": reading,
                "timestamp": datetime.now(UTC).isoformat(),
                "lat": sensor["lat"],
                "lon": sensor["lon"],
            }
            producer.send(TOPIC, value=message)
            print(f"Sent: {message}")
        time.sleep(30)


if __name__ == "__main__":
    simulate()
