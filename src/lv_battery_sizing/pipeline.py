"""Runs the whole calculation for one or more transformers."""

import logging
from dataclasses import asdict, fields

import pandas as pd

from .data_loader import DataLoader
from .models import BatteryModel, FeederModel, TrafoModel
from .preprocess import Preprocess
from .results import FeederResult

logger = logging.getLogger(__name__)


def process_trafo(
    trafo_name: str, network_path: str, data_dir: str | None = None, all_feeders: bool = False
) -> list[FeederResult]:
    """Runs the whole calculation for one transformer.

    Args:
    --------
        trafo_name:
            name of the transformer, must match the trafo name in the network and the data folder
        network_path:
            path to the pandapower network in JSON format
        data_dir:
            folder with measurement data, defaults to config.DATA_DIR
        all_feeders:
            if True, results for all feeders are returned,
            otherwise only results for feeders that need a battery
    Returns:
    --------
        one FeederResult per feeder
    """
    dl = DataLoader(trafo_name=trafo_name, data_dir=data_dir)
    pr = Preprocess(dl.load_voltage_data())
    voltage_data, undervoltage_data, trafo_needs_battery = (
        pr.preprocess_voltage_data_get_undervoltages()
    )
    if not trafo_needs_battery and not all_feeders:
        return []

    if trafo_needs_battery:
        df_vol, df_p, df_q = pr.preprocess_powers_create_pivot_tables(dl.load_power_data())
    else:
        df_vol = df_p = df_q = None
    tm = TrafoModel(voltage_data, undervoltage_data, df_vol, df_p, df_q, trafo_name, network_path)
    tm.create_and_populate_snet()

    results = []
    for feeder in tm.feeders:
        fm = FeederModel(tm, feeder)
        if trafo_needs_battery:
            fm.calculate_uv_data_and_slopes()
        else:
            fm.calculate_uv_parameters()
            fm.suitable_for_battery = False
        result = fm.result()
        if fm.suitable_for_battery:
            result.add_battery(BatteryModel(fm).calculate())
        if all_feeders or fm.suitable_for_battery:
            results.append(result)
    return results


def process_trafos(
    trafo_names: list[str],
    network_path: str,
    data_dir: str | None = None,
    all_feeders: bool = False,
) -> pd.DataFrame:
    """Runs process_trafo for every transformer, errors of single transformers are logged
    and skipped. Returns a dataframe with one row per feeder."""
    results = []
    for trafo_name in trafo_names:
        logger.info("Processing %s", trafo_name)
        try:
            results += process_trafo(trafo_name, network_path, data_dir, all_feeders)
        except Exception:
            logger.exception("Error processing transformer %s", trafo_name)
    return pd.DataFrame(
        [asdict(r) for r in results], columns=[f.name for f in fields(FeederResult)]
    )
