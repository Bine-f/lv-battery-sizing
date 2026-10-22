"""Calibration of the network model to measurements and calculation of voltage sensitivities."""

import logging

import networkx as nx
import numpy as np
import pandapower as pp
import pandapower.topology as top
import pandas as pd
import scipy.optimize as opt

from . import config
from .constants import (
    INITIAL_TRAFO_LV_KV,
    MAX_RES_FACTOR,
    MIN_RES_FACTOR,
    N_CALIBRATION_DATES,
    NOMINAL_VOLTAGE,
)
from .network import populate_snet, run_powerflow, set_volts
from .plotting import plot_feeder_volts

logger = logging.getLogger(__name__)

# Errors that make a calibration step fall back to default parameters
CALIBRATION_ERRORS = (pp.LoadflowNotConverged, ValueError, IndexError, ZeroDivisionError)


def _measured_and_simulated(volts: pd.DataFrame, bus: int) -> tuple[float, float]:
    row = volts[volts["bus"] == bus]
    if row.empty:
        raise IndexError(f"Bus {bus} has no load")
    return float(row.vol_real.values[0]), float(row.vol_pp.values[0])


def fit_res_factor_two_buses(
    snet: pp.pandapowerNet,
    min_bus: int,
    id_first: int,
    state_vol: pd.Series,
    x0: float = 1.0,
    t0: float = INITIAL_TRAFO_LV_KV,
) -> float:
    """
    Finds the line impedance factor for which the simulated voltage drop between id_first and
    min_bus equals the measured one.

    Args:
    --------
        snet:
            network in pandapower format
        min_bus:
            bus with the minimum voltage, or last bus in the feeder
        id_first:
            bus in path from transformer to min_bus with the first measured voltage
        state_vol:
            measured voltages, index are smms
        x0:
            initial guess for the impedance factor
        t0:
            transformer secondary voltage in kV
    Returns:
    --------
        impedance factor
    """

    def delta_difference(res_factor: float) -> float:
        run_powerflow(snet, res_factor=res_factor, trafo_lv=t0)
        volts = set_volts(snet, state_vol, warn=False)
        real_min, pp_min = _measured_and_simulated(volts, min_bus)
        real_first, pp_first = _measured_and_simulated(volts, id_first)
        return (real_first - real_min) - (pp_first - pp_min)

    return opt.root_scalar(delta_difference, x0=x0, x1=x0 + 0.1, method="secant").root


def fit_res_factor_feeder(
    snet: pp.pandapowerNet,
    state_vol: pd.Series,
    smms_feeder,
    x0: float = 1.0,
    t0: float = INITIAL_TRAFO_LV_KV,
) -> float:
    """Finds the line impedance factor that minimises the mean absolute difference between
    measured and simulated voltages of all smms in the feeder"""

    def mean_difference(res_f: np.ndarray) -> float:
        run_powerflow(snet, res_f[0], t0)
        volts = set_volts(snet, state_vol, warn=False)
        volts_feeder = volts.loc[volts["smm"].isin(smms_feeder)]
        return (volts_feeder.vol_pp - volts_feeder.vol_real).abs().mean()

    return opt.minimize(mean_difference, x0, method="Nelder-Mead").x[0]


def fit_trafo_lv(
    snet: pp.pandapowerNet,
    res_factor: float,
    bus: int,
    state_vol: pd.Series,
    t0: float = INITIAL_TRAFO_LV_KV,
) -> float:
    """Finds the transformer secondary voltage (kV) for which the simulated voltage at bus
    equals the measured one"""

    def difference(trafo_lv: float) -> float:
        run_powerflow(snet, res_factor=res_factor, trafo_lv=trafo_lv)
        real, simulated = _measured_and_simulated(set_volts(snet, state_vol, warn=False), bus)
        return real - simulated

    return opt.root_scalar(difference, method="secant", x0=t0, x1=t0 + 0.05, xtol=0.00002).root


def is_id_suitable(bus: int, snet: pp.pandapowerNet, volts: pd.DataFrame) -> bool:
    """A bus is suitable for calibration if it has a load with measured voltage and 3 phases"""
    phases = snet.load.loc[snet.load.bus == bus].phases.values
    return bool((volts["bus"] == bus).any()) and (len(phases) == 0 or phases[0] == 3)


def find_id_first(snet: pp.pandapowerNet, state_vol: pd.Series) -> int:
    """
    Finds the first suitable bus on the path from the transformer to the bus with the lowest
    measured voltage. A neighbour of a bus on the path is also accepted if its simulated
    voltage is very close.
    """
    run_powerflow(snet, res_factor=0.3, trafo_lv=INITIAL_TRAFO_LV_KV)
    volts = set_volts(snet, state_vol, warn=False)
    lowest_bus = volts.bus.values[0]
    tr_bus = snet.bus[snet.bus["aclass_id"] == config.TRAFO_BUS_CLASS].index[0]
    mg = top.create_nxgraph(snet, respect_switches=True)
    for bus in nx.shortest_path(mg, source=tr_bus, target=lowest_bus):
        if is_id_suitable(bus, snet, volts):
            return bus
        vol_bus = snet.res_bus.loc[bus].vm_pu
        neighbours = [n for n in mg.adj[bus] if is_id_suitable(n, snet, volts)]
        if neighbours:
            delta_vols = [abs(snet.res_bus.loc[n].vm_pu - vol_bus) for n in neighbours]
            # choose the suitable neighbour with very similar voltage as the bus in path
            if min(delta_vols) < 0.001:
                return neighbours[int(np.argmin(delta_vols))]
    raise ValueError("No suitable bus found in path from transformer to min bus")


def find_min_bus(snet: pp.pandapowerNet, state_vol: pd.Series, volts: pd.DataFrame) -> int | None:
    """Finds the bus with the lowest measured voltage that is suitable for calibration"""
    for smm in state_vol.sort_values().index:
        buses = snet.load[snet.load.smm == smm].bus.values
        if len(buses) and is_id_suitable(buses[0], snet, volts):
            return buses[0]
    return None


def _plot_state(snet, state_vol, smms_feeder, res_factor, trafo_lv, title, *buses) -> None:
    run_powerflow(snet, res_factor=res_factor, trafo_lv=trafo_lv)
    plot_feeder_volts(set_volts(snet, state_vol, warn=False), smms_feeder, *buses, title=title)


def calibrate_snet(
    snet: pp.pandapowerNet,
    state_vol: pd.Series,
    smms_feeder,
    plot: bool = False,
    x0: float = 1.0,
    t0: float = INITIAL_TRAFO_LV_KV,
    calculate_res_f: bool = True,
) -> tuple[float, float]:
    """
    Calculates the line impedance factor and transformer secondary voltage, for which the
    simulated voltages match the measured ones.

    The impedance factor is first fitted to the voltage drop between the first measured bus
    and the bus with the lowest voltage. If the result is unrealistic, it is fitted to all smms
    in the feeder. Then the transformer voltage is fitted to the voltage at the lowest bus.

    Returns:
    --------
        transformer secondary voltage in kV, impedance factor
    """
    if plot:
        _plot_state(snet, state_vol, smms_feeder, x0, t0, "Before calibration")
    run_powerflow(snet, res_factor=x0, trafo_lv=t0)
    volts = set_volts(snet, state_vol, warn=False)
    min_bus = find_min_bus(snet, state_vol, volts)
    if min_bus is None:
        logger.warning("No suitable bus for calibration, using default parameters")
        return t0, 1.0

    opt_res_f = 1.0
    if calculate_res_f and len(smms_feeder) > 2:
        try:
            id_first = find_id_first(snet, state_vol)
            opt_res_f = fit_res_factor_two_buses(snet, min_bus, id_first, state_vol, x0, t0)
        except CALIBRATION_ERRORS as e:
            logger.debug("Two bus impedance calibration failed: %s", e)
            opt_res_f = np.nan
        if not MIN_RES_FACTOR <= opt_res_f <= MAX_RES_FACTOR:
            # unrealistic or failed, calibrate using all smms in feeder
            try:
                opt_res_f = fit_res_factor_feeder(snet, state_vol, smms_feeder, x0, t0)
            except CALIBRATION_ERRORS as e:
                logger.debug("Feeder impedance calibration failed: %s", e)
                opt_res_f = 1.0
            if not np.isfinite(opt_res_f) or opt_res_f <= 0:
                opt_res_f = 1.0
        if plot:
            _plot_state(snet, state_vol, smms_feeder, opt_res_f, t0, "After res_f calibration")

    try:
        opt_trafo_lv = fit_trafo_lv(snet, opt_res_f, min_bus, state_vol, t0)
    except CALIBRATION_ERRORS as e:
        logger.warning("Transformer voltage calibration failed: %s", e)
        opt_trafo_lv = t0
    if not np.isfinite(opt_trafo_lv):
        opt_trafo_lv = t0
    if plot:
        _plot_state(
            snet, state_vol, smms_feeder, opt_res_f, opt_trafo_lv, "After trafo_lv calibration"
        )
    return opt_trafo_lv, opt_res_f


def calculate_slopes(
    snet: pp.pandapowerNet,
    battery_smms: list,
    dates,
    smms_feeder,
    df_p: pd.DataFrame,
    df_q: pd.DataFrame,
    df_vol: pd.DataFrame,
    calibrate: bool = True,
    N_of_dates: int = N_CALIBRATION_DATES,
    plot: bool = False,
) -> pd.DataFrame:
    """
    Calculates voltage sensitivities (slopes, V/kW): the voltage increase at every smm when
    the power at the battery smm is decreased by 1 kW.

    The slopes are averaged over N_of_dates evenly spaced dates. For every date the network is
    populated with measured powers and, if calibrate is True, calibrated to measured voltages.

    Args:
    --------
        snet:
            network in pandapower format
        battery_smms:
            smms for which the slopes are calculated
        dates:
            dates suitable for calibration and calculation of slopes
        smms_feeder:
            all smms in the feeder
        df_p, df_q, df_vol:
            pivot tables of measured active power, reactive power and average voltage
        calibrate:
            if True, calibrates the network before calculating slopes
        N_of_dates:
            number of dates used for calculation of slopes
        plot:
            if True, plots calibration process
    Returns:
    --------
        dataframe with battery smms as columns, all smms in the network as index
    """
    step = len(dates) // (N_of_dates + 1)
    if step == 0:
        dates_cal = list(dates)
    else:
        dates_cal = [dates[i] for i in np.arange(step, len(dates), step)[:-1]]

    slopes_smms = pd.DataFrame(index=snet.load.smm)
    for battery_smm in battery_smms:
        slopes = []
        for date in dates_cal:
            state_vol = df_vol.loc[date]
            populate_snet(snet, df_p.loc[date], df_q.loc[date], warn=False)
            opt_trafo_lv, res_f = INITIAL_TRAFO_LV_KV, 1.0
            if calibrate:
                opt_trafo_lv, res_f = calibrate_snet(
                    snet, state_vol, smms_feeder, x0=1.0, t0=INITIAL_TRAFO_LV_KV, plot=plot
                )
            # voltages with measured powers and after decreasing power at battery smm by 1 kW
            run_powerflow(snet, res_factor=res_f, trafo_lv=opt_trafo_lv)
            vol_0 = snet.res_bus.vm_pu.reindex(snet.load.bus).values
            snet.load.loc[snet.load.smm == battery_smm, "p_mw"] -= 0.001
            run_powerflow(snet, res_factor=res_f, trafo_lv=opt_trafo_lv)
            vol_1 = snet.res_bus.vm_pu.reindex(snet.load.bus).values
            slopes.append((vol_1 - vol_0) * NOMINAL_VOLTAGE)
        slopes_smms[str(battery_smm)] = np.mean(slopes, axis=0)
    return slopes_smms
