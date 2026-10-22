import numpy as np
import pandas as pd

from lv_battery_sizing.preprocess import Preprocess

TIMES = pd.date_range("2024-01-01", periods=6, freq="10min")


def voltage_frame(u_1, u_2=None, u_3=None):
    u_2 = u_1 if u_2 is None else u_2
    u_3 = u_1 if u_3 is None else u_3
    df = pd.DataFrame({"date_time": TIMES, "smm": 1, "u_1": u_1, "u_2": u_2, "u_3": u_3})
    df["u_123"] = df[["u_1", "u_2", "u_3"]].mean(axis=1)
    return df


def test_faulty_measurements_are_removed():
    df = voltage_frame(
        u_1=[230, 160, 230, 230, 230, 230.0],  # below 170 V
        u_2=[230, 230, 190, 230, 230, 230.0],  # 40 V spread
    )
    pr = Preprocess(df)
    pr.preprocess_voltages()
    assert list(pr.voltage_data.date_time) == [TIMES[0], *TIMES[3:]]


def test_single_undervoltages_are_ignored():
    df = voltage_frame(u_1=[205, 230, 205, 204, 230, 230.0])
    pr = Preprocess(df)
    voltage_data, undervoltage_data, needs_battery = pr.preprocess_voltage_data_get_undervoltages()
    assert list(undervoltage_data.date_time) == [TIMES[2], TIMES[3]]
    assert not needs_battery


def test_missing_phase_is_not_faulty():
    df = voltage_frame(u_1=[230.0] * 6, u_2=[np.nan] * 6, u_3=[np.nan] * 6)
    pr = Preprocess(df)
    pr.preprocess_voltages()
    assert len(pr.voltage_data) == 6
