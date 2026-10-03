"""Sensor noise model. sensors.csv is the single source of truth; edit it there."""

import csv
from pathlib import Path

CSV = Path(__file__).with_name("sensors.csv")


def load(path=CSV):
    """All rows of sensors.csv, keyed by channel name."""
    with Path(path).open(newline="", encoding="utf-8") as f:
        return {row["channel"]: row for row in csv.DictReader(f)}


def defaults(column, path=CSV):
    """{channel: value} of one numeric column, observed channels only."""
    return {
        name: float(row[column])
        for name, row in load(path).items()
        if row["observed"] == "yes" and float(row[column]) != 0.0
    }
