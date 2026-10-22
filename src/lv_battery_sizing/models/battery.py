import logging

import numpy as np
import pandas as pd

from ..constants import (
    INTERVALS_PER_HOUR,
    MAX_BATTERY_POWER_KW,
    MAX_BATTERY_POWER_WITH_CHARGING_KW,
    MISSING_PHASE_DEVIATION,
    UNDERVOLTAGE_LIMIT,
)
from ..preprocess import PHASES
from ..results import BatteryParameters
from .feeder import FeederModel

logger = logging.getLogger(__name__)


class BatteryModel:
    """Battery operation schedule and size needed to fix undervoltages in one feeder.

    For every timestamp the power needed at the battery smm is the voltage deficit of every
    phase of every smm divided by the voltage sensitivity (slope, V/kW) of that smm to the
    battery smm. The largest value over all smms is the battery power at that timestamp.
    """

    def __init__(self, fm: FeederModel, with_charging: bool = True, fix_all_phases: bool = True):
        self.fm = fm
        self.battery_smm = fm.battery_smm
        self.voltage_data = fm.voltage_data
        self.undervoltage_data = fm.undervoltage_data
        self.slopes = fm.slopes
        self.smms = list(fm.smms)
        self.with_charging = with_charging
        self.fix_all_phases = fix_all_phases
        self.battery_df = pd.DataFrame()

    def _phase_powers(self, missing_value: float) -> tuple[pd.DatetimeIndex, np.ndarray]:
        """Power per phase needed at the battery smm to bring every smm to the undervoltage limit.

        Returns timestamps and an array of shape (timestamps, smms, phases) in kW,
        positive values mean undervoltage. Missing measurements get missing_value as voltage
        deviation.
        """
        data = self.voltage_data.drop_duplicates(["date_time", "smm"]).set_index(
            ["date_time", "smm"]
        )
        timestamps = pd.DatetimeIndex(np.sort(self.voltage_data.date_time.unique()))
        deviation = np.stack(
            [
                (
                    UNDERVOLTAGE_LIMIT
                    - data[phase].unstack().reindex(index=timestamps, columns=self.smms)
                )
                .fillna(missing_value)
                .to_numpy()
                for phase in PHASES
            ],
            axis=-1,
        )
        slopes = self.slopes[str(self.battery_smm)].reindex(self.smms).to_numpy()
        return timestamps, deviation / slopes[None, :, None] / len(PHASES)

    def _needed_power(self, phase_powers: np.ndarray, sign: int) -> np.ndarray:
        """Largest power over smms for every timestamp. sign=1 sums the positive (discharging)
        powers of the phases, sign=-1 the negative (charging) ones."""
        if self.fix_all_phases:
            per_smm = np.where(sign * phase_powers > 0, phase_powers, 0).sum(axis=-1)
        else:
            per_smm = phase_powers.sum(axis=-1)
        return np.nan_to_num(np.nanmax(per_smm, axis=1))

    def calculate_battery_powers(self) -> None:
        """Battery schedule without recharging: power is only needed at undervoltage timestamps"""
        timestamps, phase_powers = self._phase_powers(missing_value=0.0)
        is_uv = timestamps.isin(self.undervoltage_data.date_time.unique())
        need = self._needed_power(phase_powers, sign=1)
        unrealistic = is_uv & (need >= MAX_BATTERY_POWER_KW)
        if unrealistic.any():
            logger.warning("%d unrealistic battery powers set to 0", unrealistic.sum())
        power = np.where(is_uv & ~unrealistic, need, 0.0)
        self.battery_df = pd.DataFrame({"battery_power": power}, index=timestamps)

    def calculate_battery_powers_with_charging(self) -> None:
        """Battery schedule with recharging: after an undervoltage the battery is recharged as
        fast as the voltages allow, the state of charge (kWh, <= 0) is tracked."""
        timestamps, phase_powers = self._phase_powers(missing_value=MISSING_PHASE_DEVIATION)
        is_uv = timestamps.isin(self.undervoltage_data.date_time.unique())
        discharge = self._needed_power(phase_powers, sign=1)
        charge = self._needed_power(phase_powers, sign=-1)

        powers = np.zeros(len(timestamps))
        socs = np.zeros(len(timestamps))
        soc = 0.0
        for t in range(len(timestamps)):
            if is_uv[t] or soc < 0:
                power = discharge[t] if is_uv[t] else charge[t]
                if power >= MAX_BATTERY_POWER_WITH_CHARGING_KW:
                    logger.warning("Unrealistic battery power %.1f kW set to 0", power)
                    power = 0.0
                elif not is_uv[t] and power < soc * INTERVALS_PER_HOUR:
                    # charging would overfill the battery
                    power = soc * INTERVALS_PER_HOUR
                    soc = 0.0
                else:
                    soc -= power / INTERVALS_PER_HOUR
                powers[t] = power
            socs[t] = soc
        self.battery_df = pd.DataFrame({"battery_power": powers, "soc": socs}, index=timestamps)

    def get_capacity(self) -> float:
        """Needed battery capacity in kWh"""
        if self.with_charging:
            return -self.battery_df["soc"].min()
        # largest energy of a continuous period with non-zero power
        powers = self.battery_df["battery_power"]
        segment = (powers == 0).cumsum()
        energies = powers[powers != 0].groupby(segment).sum()
        return (energies.max() if len(energies) else 0.0) / INTERVALS_PER_HOUR

    def calculate(self) -> BatteryParameters:
        """Calculates battery schedule and parameters"""
        logger.info("Calculating battery for %s", self.fm.feeder_name)
        if self.with_charging:
            self.calculate_battery_powers_with_charging()
        else:
            self.calculate_battery_powers()
        powers = self.battery_df["battery_power"]
        capacity = self.get_capacity()
        energy = powers[powers > 0].sum() / INTERVALS_PER_HOUR
        return BatteryParameters(
            smm=self.battery_smm,
            capacity_kwh=capacity,
            power_kw=powers.max(),
            cycles=energy / capacity if capacity > 0 else np.nan,
        )
