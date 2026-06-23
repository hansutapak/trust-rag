import json
import time
import random
from datetime import datetime
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers="127.0.0.1:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

SENSORS = [
    {
        "sensor_id": "EDI-AQ-001",
        "location": "Princes Street North",
        "source_type": "government_sensor",
        "metric": "AQI",
    },
    {
        "sensor_id": "EDI-TR-001",
        "location": "Princes Street Junction",
        "source_type": "council_approved",
        "metric": "traffic_density",
    },
    {
        "sensor_id": "EDI-WE-001",
        "location": "Princes Street",
        "source_type": "met_office",
        "metric": "temperature",
    },
]

TOPIC = "smart_city_sensors"


def simulate():
    while True:
        for sensor in SENSORS:
            message = {
                "sensor_id": sensor["sensor_id"],
                "location": sensor["location"],
                "source_type": sensor["source_type"],
                "metric": sensor["metric"],
                "reading": round(random.uniform(30, 80), 2),
                "timestamp": datetime.now().isoformat(),
            }
            producer.send(TOPIC, value=message)
            print(f"Sent: {message}")
        time.sleep(30)


if __name__ == "__main__":
    simulate()
