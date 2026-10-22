import os

import pandas as pd

import config


class DataLoader:
    """Loads smart meter voltage and power measurements from CSV files.

    Data for every transformer is stored in its own folder:

        <data_dir>/<trafo_name>/voltage.csv   columns: date_time, smm, u_1, u_2, u_3, u_123
        <data_dir>/<trafo_name>/power.csv     columns: date_time, smm, p, q

    Voltages are in V, active power in kW and reactive power in kvar. Files may be gzipped (.csv.gz).
    """

    def __init__(self, trafo_name=None, data_dir=None, start=None, end=None):
        self.data_dir = config.DATA_DIR if data_dir is None else data_dir
        self.trafo_name = trafo_name
        self.voltage_data = None
        self.power_data = None
        self.start = start
        self.end = end

    def _file_path(self, file_name, trafo_name=None):
        """Returns path to file_name in the trafo folder, gzipped files (.csv.gz) are also accepted"""
        path = os.path.join(self.data_dir, trafo_name or self.trafo_name, file_name)
        if not os.path.isfile(path) and os.path.isfile(path + ".gz"):
            path += ".gz"
        return path

    def _read_csv(self, file_name):
        df = pd.read_csv(self._file_path(file_name), parse_dates=["date_time"])
        if self.start is not None:
            df = df[df.date_time >= pd.Timestamp(self.start)]
        if self.end is not None:
            df = df[df.date_time < pd.Timestamp(self.end)]
        return df.sort_values("date_time").reset_index(drop=True)

    def load_trafo_data(self):
        """Loads power and voltage data for all smms of given trafo network"""
        self.load_voltage_data()
        self.load_power_data()
        return self.voltage_data, self.power_data

    def load_voltage_data(self):
        """Loads voltage data for all smms of given trafo network"""
        self.voltage_data = self._read_csv(config.VOLTAGE_FILE)
        return self.voltage_data

    def load_power_data(self):
        """Loads power data for all smms of given trafo network"""
        self.power_data = self._read_csv(config.POWER_FILE)
        return self.power_data

    def list_trafos(self):
        """Returns names of all transformers with voltage data in data_dir"""
        return sorted(
            name
            for name in os.listdir(self.data_dir)
            if os.path.isfile(self._file_path(config.VOLTAGE_FILE, name))
        )

    def find_trafo_candidates(self, min_events=4):
        """Creates a list of trafos that have undervoltages, and could be suitable for battery installation.
        Returns a list of trafos that have at least min_events undervoltage events with at least 20 min of
        undervoltages in a row, and data is not faulty"""
        candidates = []
        for trafo_name in self.list_trafos():
            self.trafo_name = trafo_name
            df = self.load_voltage_data()
            phases = df[["u_1", "u_2", "u_3"]]
            # missing phases are not considered undervoltage
            low = (phases.fillna(208) < 207).any(axis=1)
            min_u = phases.min(axis=1)
            valid = ((phases.max(axis=1) - min_u) < 30) & (min_u > 170)
            events = pd.Index(df.loc[low & valid, "date_time"].unique())
            low_times = pd.Index(df.loc[low, "date_time"].unique())
            # an event counts if there is another undervoltage in the network 10 minutes before or after
            step = pd.Timedelta("10min")
            consecutive = (events + step).isin(low_times) | (events - step).isin(low_times)
            n_events = int(consecutive.sum())
            if n_events >= min_events:
                candidates.append((trafo_name, n_events))
        self.trafo_name = None
        candidates.sort(key=lambda c: c[1], reverse=True)
        return [name for name, _ in candidates]
