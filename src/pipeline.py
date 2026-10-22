import pandas as pd

from data_loader import DataLoader
from preprocess import Preprocess
from models.trafo_model import TrafoModel
from models.feeder_model import FeederModel
from models.battery_model import BatteryModel


def process_trafo(trafo_name, network_path, data_dir=None, all_feeders=False):
    """Runs the whole calculation for one transformer.

    Args:
    --------
        trafo_name: str
            name of the transformer, must match the trafo name in the network and the data folder name
        network_path: str
            path to the pandapower network in JSON format
        data_dir: str
            folder with measurement data, defaults to config.DATA_DIR
        all_feeders: bool
            if True, results for all feeders are returned,
            otherwise only results for feeders suitable for battery
    Returns:
    --------
        pd.DataFrame with one row per feeder
    """
    res = pd.DataFrame()
    dl = DataLoader(trafo_name=trafo_name, data_dir=data_dir)
    voltage_data = dl.load_voltage_data()
    pr = Preprocess(voltage_data)
    voltage_data, undervoltage_data, trafo_suitable_for_battery = pr.preprocess_voltage_data_get_undervoltages()

    if trafo_suitable_for_battery:
        # There are undervoltages, we need to fix
        power_data = dl.load_power_data()
        df_vol, df_p, df_q = pr.preprocess_powers_create_pivot_tables(
            power_data)
        tm = TrafoModel(voltage_data, undervoltage_data, df_vol, df_p, df_q,
                        trafo_name, network_path)
        tm.create_and_populate_snet()
        for feeder in tm.feeders:
            fm = FeederModel(tm, feeder)
            fm.calculate_uv_data_and_slopes()
            if all_feeders or fm.suitable_for_battery:
                bm = BatteryModel(fm)
                bm.calculate_battery_parameters()
                res = pd.concat([res, fm.feeder_res], ignore_index=True)
    elif all_feeders:
        tm = TrafoModel(voltage_data, undervoltage_data, None, None, None,
                        trafo_name, network_path)
        tm.create_and_populate_snet()
        for feeder in tm.feeders:
            fm = FeederModel(tm, feeder)
            fm.calculate_and_write_uv_data(empty_battery_columns=True)
            res = pd.concat([res, fm.feeder_res], ignore_index=True)
    return res


def process_trafos(trafo_names, network_path, data_dir=None, all_feeders=False):
    """Runs process_trafo for every transformer, errors for single transformers are reported and skipped"""
    res = pd.DataFrame()
    for trafo_name in trafo_names:
        print(trafo_name)
        try:
            trafo_res = process_trafo(trafo_name, network_path, data_dir,
                                      all_feeders)
            res = pd.concat([res, trafo_res], ignore_index=True)
        except Exception as e:
            print(f"Error processing transformer {trafo_name}: {e}")
    return res
