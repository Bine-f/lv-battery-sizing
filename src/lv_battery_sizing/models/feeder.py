import numpy as np
import pandas as pd

from ..calibration import calculate_slopes
from ..constants import (
    CALIBRATION_QUANTILE,
    MIN_UV_TIMESTAMPS,
    NOMINAL_VOLTAGE,
    UNDERVOLTAGE_LIMIT_PU,
)
from ..network import get_feeder_smms
from ..preprocess import get_data_from_smm_list, limit_voltage
from ..results import FeederResult
from .trafo import TrafoModel


def order_smms_by_undervoltage_sum(uv_data_avg: pd.DataFrame, vol_lim: float) -> list:
    """Orders smms by the sum of their voltage deficits below vol_lim, largest first"""
    smms = uv_data_avg.smm.unique()
    sums = [(vol_lim - uv_data_avg[uv_data_avg.smm == smm]["avg_u"]).sum() for smm in smms]
    return [smms[i] for i in np.argsort(sums)][::-1]


class FeederModel:
    """Undervoltage analysis and voltage sensitivities of one feeder"""

    def __init__(self, tm: TrafoModel, feeder_name: str):
        self.tm = tm
        self.feeder_name = feeder_name
        self.snet = tm.snet
        self.smms = get_feeder_smms(self.snet, feeder_name)
        self.voltage_data = get_data_from_smm_list(tm.voltage_data, self.smms)
        self.undervoltage_data = get_data_from_smm_list(tm.undervoltage_data, self.smms)
        self.battery_smm = None
        self.suitable_for_battery = False
        self.slopes = None
        self.enough_voltage_data = tm.enough_voltage_data

    def define_calibration_lim_vol(self) -> None:
        """Defines the calibration voltage limit, so that we have enough low voltage data"""
        min_voltages = self.voltage_data.groupby("date_time")["u_123"].min()
        self.lim_vol_avg = max(
            min_voltages.quantile(CALIBRATION_QUANTILE) / NOMINAL_VOLTAGE, UNDERVOLTAGE_LIMIT_PU
        )

    def define_and_limit_voltage(self, average: bool = True) -> None:
        """Selects timestamps with low average voltages for calibration"""
        self.define_calibration_lim_vol()
        self.uv_data_avg = limit_voltage(
            self.voltage_data, lim_vol=self.lim_vol_avg, average=average
        )
        self.avg_dates = self.uv_data_avg["date_time"].unique()

    def calculate_uv_parameters(self) -> None:
        self.N_of_UV = len(self.undervoltage_data)
        self.N_of_smms = len(self.smms)
        self.N_of_uv_smms = self.undervoltage_data["smm"].nunique()
        self.N_dates = self.undervoltage_data["date_time"].nunique()
        self.suitable_for_battery = self.N_dates >= MIN_UV_TIMESTAMPS

    def result(self) -> FeederResult:
        """Undervoltage statistics of the feeder"""
        return FeederResult(
            trafo=self.tm.trafo_name,
            feeder=self.feeder_name,
            n_undervoltages=self.N_of_UV,
            n_meters=self.N_of_smms,
            n_meters_with_uv=self.N_of_uv_smms,
            n_uv_timestamps=self.N_dates,
        )

    def determine_battery_smm(self) -> None:
        """Places the battery at the smm with the largest sum of undervoltages
        that exists in the network model"""
        ordered = order_smms_by_undervoltage_sum(self.uv_data_avg, vol_lim=self.lim_vol_avg)
        in_network = [smm for smm in ordered if smm in self.snet.load.smm.values]
        if not in_network:
            raise ValueError(f"No battery smm found for feeder {self.feeder_name}")
        self.battery_smm = in_network[0]

    def calculate_slopes(self) -> None:
        """Calculates voltage sensitivities of all smms in the feeder to the battery smm"""
        self.define_and_limit_voltage()
        self.determine_battery_smm()
        self.slopes = calculate_slopes(
            self.snet,
            [self.battery_smm],
            self.avg_dates,
            self.smms,
            self.tm.df_p,
            self.tm.df_q,
            self.tm.df_vol,
            calibrate=self.enough_voltage_data,
        )

    def calculate_uv_data_and_slopes(self) -> None:
        """Calculates undervoltage parameters and, if a battery is needed, voltage sensitivities"""
        self.calculate_uv_parameters()
        if self.suitable_for_battery:
            self.calculate_slopes()
