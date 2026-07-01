"""Check if the bayesian credibility updates work correctly."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scoring.trs import get_bayesian_credibility, update_bayesian_credibility

SENSORS = [
    ("government_sensor", "BCN-AQ-001"),
    ("citizen_device", "BCN-SM-001"),
    ("commercial_iot", "BCN-WB-001"),
]


def main():
    print("Bayesian credibility verification\n")

    # Initial credibility(match static values)
    print("Initial (prior) credibility")
    for stype, sid in SENSORS:
        cred = get_bayesian_credibility(stype, sid)
        print(f"{stype:20s} {cred:.4f}")

    # Simulate accurate readings for citizen device
    print("\nSimulating 10 accurate readings for citizen_device ...")
    for _ in range(10):
        update_bayesian_credibility("citizen_device", "BCN-SM-001", accurate=True)
    print(
        f"  citizen_device  after 10 good: {get_bayesian_credibility('citizen_device', 'BCN-SM-001'):.4f}"
    )

    # Simulate inaccurate readings for government_sensor
    print("Simulating 5 inaccurate readings for government_sensor ...")
    for _ in range(5):
        update_bayesian_credibility("government_sensor", "BCN-AQ-001", accurate=False)
    print(
        f"  government_sensor after 5 bad: {get_bayesian_credibility('government_sensor', 'BCN-AQ-001'):.4f}"
    )

    # Show different sensor of the same type is unaffected
    print("\nOther sensor of same type (BCN-AQ-002) still uses the prior:")
    print(
        f"  government_sensor  {get_bayesian_credibility('government_sensor', 'BCN-AQ-002'):.4f}"
    )

    print("\nIf initial values match static rules and they change after updates,")
    print("Bayesian credibility is working.")


if __name__ == "__main__":
    main()
