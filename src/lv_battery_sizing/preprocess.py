"""Cleaning of measurements, undervoltage detection and resampling to pivot tables."""

import logging

import numpy as np
import pandas as pd

from .constants import (
    MAX_PHASE_ASYMMETRY,
    MAX_PHASE_SPREAD,
    MEASUREMENT_INTERVAL,
    MIN_UV_SAMPLES_FOR_ASYMMETRY_CHECK,
    MIN_UV_TIMESTAMPS,
    MIN_VALID_VOLTAGE,
    NOMINAL_VOLTAGE,
    UNDERVOLTAGE_LIMIT_PU,
)

logger = logging.getLogger(__name__)

PHASES = ["u_1", "u_2", "u_3"]


def limit_voltage(
    vol_df: pd.DataFrame, lim_vol: float = UNDERVOLTAGE_LIMIT_PU, average: bool = False
) -> pd.DataFrame:
    """Returns rows where the minimal (or average) phase voltage in pu is at most lim_vol"""
    column = "avg_u" if average else "min_u"
    return vol_df[vol_df[column] <= lim_vol]


def get_data_from_smm_list(df: pd.DataFrame, smm_list) -> pd.DataFrame:
    return df[df["smm"].isin(smm_list)]


class Preprocess:
    def __init__(self, voltage_data: pd.DataFrame, power_data: pd.DataFrame | None = None):
        self.voltage_data = voltage_data
        self.power_data = power_data
        self.undervoltage_data: pd.DataFrame | None = None
        self.minimal_vol = MIN_VALID_VOLTAGE / NOMINAL_VOLTAGE
        self.max_diff = MAX_PHASE_SPREAD / NOMINAL_VOLTAGE
        self.fillna_method: str | None = None  # None, "ffill" or "bfill"
        self.df_vol: pd.DataFrame | None = None
        self.df_p: pd.DataFrame | None = None
        self.df_q: pd.DataFrame | None = None

    def crop_voltage_data(self) -> None:
        """Crops voltage data to the last year"""
        self.end_date = self.voltage_data.date_time.max()
        self.start_date = self.end_date - pd.Timedelta("365 days")
        self.voltage_data = self.voltage_data[self.voltage_data.date_time >= self.start_date]

    def crop_to_one_year(self) -> None:
        """Crops voltage and power data to the last year of voltage data"""
        self.crop_voltage_data()
        self.power_data = self.power_data[self.power_data.date_time >= self.start_date]

    def _pivot(self, data: pd.DataFrame, values: str) -> pd.DataFrame:
        """Pivots data to a timestamps x smms table and aligns it to the measurement interval"""
        pivoted = data.pivot_table(index="date_time", columns="smm", values=values, aggfunc="mean")
        pivoted.index = pd.to_datetime(pivoted.index)
        if self.fillna_method is not None:
            pivoted = getattr(pivoted, self.fillna_method)()
        half_interval = pd.Timedelta(MEASUREMENT_INTERVAL) / 2
        pivoted = pivoted.resample(half_interval).bfill()
        pivoted.index = pivoted.index - half_interval
        return pivoted.resample(MEASUREMENT_INTERVAL, label="right").mean()

    def create_pivot_tables(self) -> None:
        """Creates pivot tables of average voltage, active and reactive power"""
        self.df_vol = self._pivot(self.voltage_data, "u_123")
        self.df_p = self._pivot(self.power_data, "p")
        self.df_q = self._pivot(self.power_data, "q")

    def preprocess_powers_create_pivot_tables(
        self, power_data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Crops power data and creates pivot tables of voltages and powers"""
        self.power_data = power_data
        self.crop_to_one_year()
        self.create_pivot_tables()
        return self.df_vol, self.df_p, self.df_q

    def preprocess_voltages(self) -> None:
        """Removes faulty voltage measurements.

        A measurement is faulty if any phase is below minimal_vol or if the difference
        between phases is above max_diff.
        """
        df = self.voltage_data.copy()
        df["min_u"] = df[PHASES].min(axis=1) / NOMINAL_VOLTAGE
        df["avg_u"] = df[PHASES].mean(axis=1) / NOMINAL_VOLTAGE
        df["max_u"] = df[PHASES].max(axis=1) / NOMINAL_VOLTAGE
        df["diff_u"] = df["max_u"] - df["min_u"]
        df = df[(df["min_u"] >= self.minimal_vol) & (df["diff_u"] <= self.max_diff)]
        self.voltage_data = df.sort_values(by=["smm", "date_time"])

    def get_undervoltage_data(
        self, lim_vol: float = UNDERVOLTAGE_LIMIT_PU, remove_single_occurences: bool = True
    ) -> None:
        """Finds undervoltages in voltage data

        Args:
        --------
            lim_vol:
               voltage in pu, below which we consider it undervoltage
            remove_single_occurences:
                if True, undervoltage must occur at least twice in a row
        """
        uv = limit_voltage(self.voltage_data.copy(), lim_vol=lim_vol, average=False)
        uv = uv.sort_values(by=["smm", "date_time"])
        uv["timedelta"] = uv.groupby("smm")["date_time"].diff()
        if remove_single_occurences:
            step = pd.Timedelta(MEASUREMENT_INTERVAL)
            uv = uv[(uv["timedelta"] == step) | (uv["timedelta"].shift(-1) == step)]
        self.undervoltage_data = uv

    def is_trafo_suitable_for_battery(self) -> bool:
        """Returns True if there are undervoltages at enough timestamps"""
        if self.undervoltage_data is None:
            return False
        return self.undervoltage_data["date_time"].nunique() >= MIN_UV_TIMESTAMPS

    def remove_smm(self, smm) -> None:
        """Removes smm from voltage and undervoltage data"""
        self.voltage_data = self.voltage_data[self.voltage_data.smm != smm]
        self.undervoltage_data = self.undervoltage_data[self.undervoltage_data.smm != smm]

    def remove_asymmetric_smms(self) -> None:
        """Removes smms with many undervoltages whose phases are too asymmetric to be real"""
        smm_counts = self.undervoltage_data["smm"].value_counts()
        for smm in smm_counts[smm_counts > MIN_UV_SAMPLES_FOR_ASYMMETRY_CHECK].index:
            smm_data = self.voltage_data[self.voltage_data["smm"] == smm]
            averages = np.sort([np.average(smm_data[phase]) for phase in PHASES])
            # difference between average of the two highest phases and the lowest phase
            if (averages[1] + averages[2]) / 2 - averages[0] > MAX_PHASE_ASYMMETRY:
                self.remove_smm(smm)
                logger.info("Removed smm %s with asymmetric voltages", smm)

    def preprocess_voltage_data_get_undervoltages(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
        """Cleans voltage data, finds undervoltages and determines if trafo needs a battery"""
        self.crop_voltage_data()
        self.preprocess_voltages()
        self.get_undervoltage_data()
        self.remove_asymmetric_smms()
        return self.voltage_data, self.undervoltage_data, self.is_trafo_suitable_for_battery()
