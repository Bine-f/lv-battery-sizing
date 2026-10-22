"""Plots for inspecting calibration and the network (used interactively)."""

import matplotlib.pyplot as plt
import numpy as np
import pandapower.plotting as plot

from .network import get_bus_from_smm, run_powerflow


def plot_volts(volts, min_bus=None, id_first=None, title=None):
    plt.plot(list(volts["vol_pp"]), label="Simulated voltage")
    plt.plot(list(volts["vol_real"]), label="Measured voltage")
    if min_bus is not None:
        min_ind = np.where(volts.bus == min_bus)[0][0]
        plt.axvline(x=min_ind, color="r", linestyle="--", label="Min bus")
    if id_first is not None:
        first_ind = np.where(volts.bus == id_first)[0][0]
        plt.axvline(x=first_ind, color="g", linestyle="--", label="First bus in path")
    plt.legend()
    if title is not None:
        plt.title(title)
    plt.show()


def plot_feeder_volts(volts, smms_feeder, min_bus=None, id_first=None, title=None):
    volts_feeder = volts.loc[volts.smm.isin(smms_feeder)]
    plot_volts(volts_feeder, min_bus, id_first, title=title)


def show_busses(snet, busses):
    "Plots network with busses highlighted"
    snet.load.p_mw = 0
    run_powerflow(snet)
    snet.res_bus.vm_pu = 0.9
    snet.res_bus.loc[busses, "vm_pu"] = 1.1
    plot.pf_res_plotly(snet)


def show_smms(snet, smms):
    "Plots network with smms highlighted"
    busses = [get_bus_from_smm(snet, smm) for smm in smms]
    show_busses(snet, busses)
