import pandas as pd

from lv_battery_sizing.data_loader import DataLoader


def test_reads_gzipped_csv_and_filters_period(tmp_path):
    folder = tmp_path / "T1"
    folder.mkdir()
    df = pd.DataFrame(
        {
            "date_time": pd.date_range("2024-01-01", periods=4, freq="10min"),
            "smm": 1,
            "p": [1.0, 2.0, 3.0, 4.0],
            "q": 0.0,
        }
    )
    df.to_csv(folder / "power.csv.gz", index=False)
    dl = DataLoader("T1", data_dir=tmp_path, start="2024-01-01 00:10", end="2024-01-01 00:30")
    assert list(dl.load_power_data().p) == [2.0, 3.0]


def test_list_trafos_only_returns_folders_with_voltage_data(tmp_path):
    (tmp_path / "A").mkdir()
    (tmp_path / "A" / "voltage.csv").write_text("date_time,smm,u_1,u_2,u_3,u_123\n")
    (tmp_path / "B").mkdir()
    assert DataLoader(data_dir=tmp_path).list_trafos() == ["A"]
