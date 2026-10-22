from fastapi.testclient import TestClient

from lv_battery_sizing.api import app


def test_get_results_returns_json_with_nulls(in_repo_root):
    response = TestClient(app).get("/get_results/", params={"all_feeders": True})
    assert response.status_code == 200
    records = response.json()
    assert {r["feeder"] for r in records} == {
        "FEEDER-1 TRAFO_A",
        "FEEDER-2 TRAFO_A",
        "FEEDER-3 TRAFO_A",
    }
    without_battery = [r for r in records if r["feeder"] != "FEEDER-2 TRAFO_A"]
    assert all(r["battery_capacity_kwh"] is None for r in without_battery)
