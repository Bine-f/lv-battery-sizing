"""Project configuration.

Paths default to the bundled synthetic demo data, they can be overridden
with the environment variables below or with command line arguments of main.py.
"""

import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# pandapower network (JSON) containing all transformer areas
NETWORK_PATH = os.environ.get(
    "LVBAT_NETWORK_PATH", os.path.join(ROOT_DIR, "data", "demo", "network.json")
)

# Folder with one sub-folder per transformer, each containing voltage.csv and power.csv
DATA_DIR = os.environ.get("LVBAT_DATA_DIR", os.path.join(ROOT_DIR, "data", "demo"))

VOLTAGE_FILE = "voltage.csv"
POWER_FILE = "power.csv"

# Network conventions
# value of bus["aclass_id"] that marks the LV busbar of the transformer
TRAFO_BUS_CLASS = "TR"
# every path from the transformer to a load contains exactly one line whose
# name contains this tag, the name of that line is used as the feeder name
FEEDER_LINE_TAG = "FEEDER"
