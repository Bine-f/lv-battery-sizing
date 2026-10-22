"""Paths and network conventions.

Paths default to the synthetic demo data in ./data/demo and can be overridden with
environment variables or command line arguments.
"""

import os

NETWORK_PATH = os.environ.get("LVBAT_NETWORK_PATH", os.path.join("data", "demo", "network.json"))
# Folder with one sub-folder per transformer, each containing voltage.csv and power.csv
DATA_DIR = os.environ.get("LVBAT_DATA_DIR", os.path.join("data", "demo"))

VOLTAGE_FILE = "voltage.csv"
POWER_FILE = "power.csv"

# value of bus["aclass_id"] that marks the LV busbar of the transformer
TRAFO_BUS_CLASS = "TR"
# every path from the transformer to a load contains exactly one line whose
# name contains this tag, the name of that line is used as the feeder name
FEEDER_LINE_TAG = "FEEDER"
