import os

import pytest

from data_loader import DataLoader
from pipeline import process_trafos

ROOT = os.path.join(os.path.dirname(__file__), "..")
DEMO_DIR = os.path.join(ROOT, "data", "demo")
NETWORK = os.path.join(DEMO_DIR, "network.json")


def test_candidates():
    assert DataLoader(data_dir=DEMO_DIR).find_trafo_candidates() == ["TRAFO_A"]


def test_demo_battery_result():
    res = process_trafos(["TRAFO_A"], NETWORK, DEMO_DIR)
    assert len(res) == 1
    row = res.iloc[0]
    assert row["Feeder"] == "FEEDER-2 TRAFO_A"
    assert row["battery_smm"] == 1022
    assert row["battery_capacity"] == pytest.approx(0.987918, rel=1e-6)
    assert row["battery_power"] == pytest.approx(4.645883, rel=1e-6)
    assert row["battery_cycles"] == pytest.approx(2.243805, rel=1e-6)
