import networkx as nx
import pandapower.topology as top
import pandas as pd

from .. import config
from ..constants import MIN_VOLTAGE_DATA_SHARE
from ..network import load_network, trafo_area


class TrafoModel:
    """Network model and measurements of one transformer area"""

    def __init__(
        self,
        voltage_data: pd.DataFrame | None,
        undervoltage_data: pd.DataFrame | None,
        df_vol: pd.DataFrame | None,
        df_p: pd.DataFrame | None,
        df_q: pd.DataFrame | None,
        trafo_name: str,
        network_path: str,
    ):
        self.voltage_data = voltage_data
        self.undervoltage_data = undervoltage_data
        self.df_vol = df_vol
        self.df_p = df_p
        self.df_q = df_q
        self.trafo_name = trafo_name
        self.network_path = network_path
        self.feeders = None
        self.snet = None
        if self.voltage_data is not None:
            self.enough_voltage_data = self.is_there_enough_voltage_data()

    def create_snet(self) -> None:
        """Creates pandapower network of the transformer area"""
        self.snet = trafo_area(load_network(self.network_path), self.trafo_name)

    def populate_snet_feeders(self) -> None:
        """Adds feeder column to snet.load based on the path from transformer to bus"""
        tr_bus = self.snet.bus[self.snet.bus["aclass_id"] == config.TRAFO_BUS_CLASS].index[0]
        mg = top.create_nxgraph(self.snet, respect_switches=True)
        feeders = []
        for bus in self.snet.load.bus:
            pth = nx.shortest_path(mg, source=tr_bus, target=bus)
            lines = self.snet.line[
                self.snet.line["from_bus"].isin(pth) & self.snet.line["to_bus"].isin(pth)
            ]
            # Every path from trafo to bus contains a line tagged with FEEDER_LINE_TAG,
            # its name is the feeder name
            feeder_lines = lines[lines.name.str.contains(config.FEEDER_LINE_TAG, regex=False)]
            feeders.append(feeder_lines.name.values[0])
        self.snet.load["feeder"] = feeders

    def populate_snet_phases(self) -> None:
        """Adds phases column to snet.load: 1 if a phase is missing in more than half of the
        measurements, 3 otherwise, 0 if there is no voltage data"""
        if self.voltage_data is None:
            self.snet.load["phases"] = 0
            return
        phs = []
        for smm in self.snet.load.smm:
            df_vol_smm = self.voltage_data.loc[self.voltage_data["smm"] == smm]
            half = len(df_vol_smm) // 2
            one_phase = df_vol_smm.u_2.isna().sum() > half or df_vol_smm.u_3.isna().sum() > half
            phs.append(1 if one_phase else 3)
        self.snet.load["phases"] = phs

    def create_and_populate_snet(self) -> None:
        """Creates network of the transformer area and assigns feeders and phases to loads"""
        if self.snet is None:
            self.create_snet()
        self.populate_snet_feeders()
        self.populate_snet_phases()
        self.feeders = self.snet.load.feeder.unique()

    def percentage_of_voltage_data(self) -> float:
        """Share of smms in the network for which we have voltage data"""
        return self.voltage_data.smm.nunique() / self.snet.load.smm.nunique()

    def is_there_enough_voltage_data(self) -> bool:
        if self.snet is None:
            self.create_snet()
        return self.percentage_of_voltage_data() > MIN_VOLTAGE_DATA_SHARE
