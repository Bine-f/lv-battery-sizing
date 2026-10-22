import pytest
from conftest import DEMO_DIR, NETWORK

from lv_battery_sizing.data_loader import DataLoader
from lv_battery_sizing.pipeline import process_trafos


def test_candidates():
    assert DataLoader(data_dir=DEMO_DIR).find_trafo_candidates() == ["TRAFO_A"]


def test_demo_battery_result():
    res = process_trafos(["TRAFO_A"], NETWORK, DEMO_DIR)
    assert len(res) == 1
    row = res.iloc[0]
    assert row["feeder"] == "FEEDER-2 TRAFO_A"
    assert row["battery_smm"] == 1022
    assert row["battery_capacity_kwh"] == pytest.approx(0.987918, rel=1e-6)
    assert row["battery_power_kw"] == pytest.approx(4.645883, rel=1e-6)
    assert row["battery_cycles"] == pytest.approx(2.243805, rel=1e-6)


def test_all_feeders_includes_feeders_without_battery():
    res = process_trafos(["TRAFO_A", "TRAFO_B"], NETWORK, DEMO_DIR, all_feeders=True)
    assert len(res) == 5
    assert res["battery_smm"].notna().sum() == 1
