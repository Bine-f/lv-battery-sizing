from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lv_battery_sizing.models import BatteryModel

TIMES = pd.date_range("2024-01-01 18:00", periods=4, freq="10min")


def feeder(voltages):
    """One meter (smm 1) with equal phase voltages and a sensitivity of 1 V/kW"""
    voltage_data = pd.DataFrame(
        {"date_time": TIMES, "smm": 1, "u_1": voltages, "u_2": voltages, "u_3": voltages}
    )
    return SimpleNamespace(
        feeder_name="F1",
        battery_smm=1,
        smms=pd.Series([1]),
        voltage_data=voltage_data,
        undervoltage_data=voltage_data[voltage_data.u_1 < 207],
        slopes=pd.DataFrame({"1": [1.0]}, index=[1]),
    )


def test_discharge_and_recharge():
    # 3 V deficit per phase at 18:10 needs 3 kW for 10 min = 0.5 kWh,
    # at 18:20 there is 6 V headroom per phase, recharging is limited by the empty capacity
    bm = BatteryModel(feeder([210.0, 204.0, 213.0, 210.0]))
    params = bm.calculate()
    np.testing.assert_allclose(bm.battery_df["battery_power"], [0, 3, -3, 0])
    np.testing.assert_allclose(bm.battery_df["soc"], [0, -0.5, 0, 0])
    assert params.capacity_kwh == pytest.approx(0.5)
    assert params.power_kw == pytest.approx(3)
    assert params.cycles == pytest.approx(1)


def test_without_charging_capacity_is_largest_continuous_energy():
    bm = BatteryModel(feeder([204.0, 204.0, 210.0, 201.0]), with_charging=False)
    params = bm.calculate()
    np.testing.assert_allclose(bm.battery_df["battery_power"], [3, 3, 0, 6])
    assert params.capacity_kwh == pytest.approx(1.0)
    assert params.power_kw == pytest.approx(6)


def test_missing_phases_do_not_limit_charging():
    fm = feeder([210.0, 204.0, 207.0, 210.0])
    fm.voltage_data.loc[2, ["u_2", "u_3"]] = np.nan
    bm = BatteryModel(fm)
    bm.calculate()
    # phase 1 is exactly at the limit and allows no charging, missing phases are ignored
    np.testing.assert_allclose(bm.battery_df["battery_power"], [0, 3, 0, -3])


def test_timestamps_without_measurements_do_not_limit_charging():
    fm = feeder([210.0, 204.0, 213.0, 210.0])
    fm.voltage_data = fm.voltage_data.drop(index=2)
    fm.voltage_data.loc[2] = [TIMES[2], 1, np.nan, np.nan, np.nan]
    bm = BatteryModel(fm)
    bm.calculate()
    np.testing.assert_allclose(bm.battery_df["battery_power"], [0, 3, -3, 0])
