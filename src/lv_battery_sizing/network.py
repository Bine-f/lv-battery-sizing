"""Helpers for the pandapower network model: loading, transformer areas and power flows."""

import logging

import numpy as np
import pandapower as pp
import pandapower.topology as top
import pandas as pd
from pandapower.toolbox import select_subnet

from .constants import NOMINAL_VOLTAGE

logger = logging.getLogger(__name__)


def load_network(json_path: str) -> pp.pandapowerNet:
    return pp.from_json(json_path)


def area_of_bus(net: pp.pandapowerNet, bus: int) -> pp.pandapowerNet:
    """Returns the connected part of the network that contains bus"""
    if bus not in net.bus.index:
        raise ValueError(f"Bus {bus} does not exist in the network")
    buses = top.connected_component(top.create_nxgraph(net), bus)
    return select_subnet(net, list(buses))


def trafo_area(net: pp.pandapowerNet, trafo_name: str) -> pp.pandapowerNet:
    """Returns the network supplied by the transformer named trafo_name. If no name matches
    exactly, the transformer whose name contains trafo_name is used."""
    trafos = net.trafo.loc[net.trafo.name == trafo_name]
    if trafos.empty:
        trafos = net.trafo.loc[net.trafo.name.str.contains(trafo_name, regex=False, na=False)]
    if len(trafos) != 1:
        raise ValueError(
            f"Expected one transformer named '{trafo_name}' in the network, found {len(trafos)}"
        )
    return area_of_bus(net, trafos.lv_bus.values[0])


def get_feeder_smms(snet: pp.pandapowerNet, feeder: str) -> pd.Series:
    """Returns smms in given feeder"""
    return snet.load[snet.load["feeder"] == feeder].smm


def get_bus_from_smm(snet: pp.pandapowerNet, smm) -> int:
    return snet.load.loc[snet.load.smm == smm].bus.values[0]


def get_smm_from_bus(snet: pp.pandapowerNet, bus: int):
    return snet.load.loc[snet.load.bus == bus].smm.values[0]


def set_volts(
    snet: pp.pandapowerNet, state_vol: pd.Series, warn: bool = True, sort: bool = True
) -> pd.DataFrame:
    """Creates a dataframe with measured voltages and power flow results for every load

    Args:
    --------
        snet:
            network in pandapower format, with power flow results
        state_vol:
            average measured voltages in V, index are smms
        warn:
            if True, logs smms without measured voltage
        sort:
            if True, sorts the dataframe by measured voltage
    Returns:
    --------
        dataframe with columns bus, p_mw, q_mvar, smm, name, vol_pp and vol_real (both in pu)
    """
    volts = snet.load[["bus", "p_mw", "q_mvar", "smm", "name"]].copy()
    volts["vol_pp"] = snet.res_bus.vm_pu.reindex(volts.bus).values
    volts["vol_real"] = volts.smm.map(state_vol).astype(float) / NOMINAL_VOLTAGE
    if warn:
        for smm in volts.loc[volts.vol_real.isna(), "smm"]:
            logger.warning("Missing measured voltage for smm %s", smm)
    if sort:
        volts.sort_values(by=["vol_real"], ascending=True, inplace=True)
    return volts


def run_powerflow(snet: pp.pandapowerNet, res_factor: float = 1.0, trafo_lv: float = 0.4) -> None:
    """Runs a power flow with line impedances scaled by res_factor and the transformer
    secondary voltage set to trafo_lv (kV). The original network parameters are always restored.
    """
    if not np.isfinite(res_factor) or not np.isfinite(trafo_lv):
        raise ValueError("res_factor or trafo_lv is not finite")
    orig_lv = snet.trafo["vn_lv_kv"].copy()
    orig_r = snet.line["r_ohm_per_km"].copy()
    orig_x = snet.line["x_ohm_per_km"].copy()
    snet.line["r_ohm_per_km"] = orig_r * res_factor
    snet.line["x_ohm_per_km"] = orig_x * res_factor
    snet.trafo["vn_lv_kv"] = trafo_lv
    try:
        pp.runpp(snet, numba=False)
    finally:
        snet.trafo["vn_lv_kv"] = orig_lv
        snet.line["r_ohm_per_km"] = orig_r
        snet.line["x_ohm_per_km"] = orig_x


def populate_snet(
    snet: pp.pandapowerNet, state_p: pd.Series, state_q: pd.Series, warn: bool = True
) -> None:
    """Sets load powers from measurements, missing values are set to 0

    Args:
    --------
        snet:
            network in pandapower format
        state_p:
            measured active powers in kW, index are smms
        state_q:
            measured reactive powers in kvar, index are smms
        warn:
            if True, logs smms without measured power
    """
    p = snet.load.smm.map(state_p).astype(float)
    q = snet.load.smm.map(state_q).astype(float)
    if warn:
        for smm in snet.load.smm[p.isna()]:
            logger.warning("Missing active power for smm %s", smm)
        for smm in snet.load.smm[q.isna()]:
            logger.warning("Missing reactive power for smm %s", smm)
    snet.load["p_mw"] = p.fillna(0) / 1000
    snet.load["q_mvar"] = q.fillna(0) / 1000
