"""Generates a synthetic LV network and smart meter measurements for the demo.

Two transformer areas are created:

    TRAFO_A - three feeders, one of them long and weak, evening peaks cause undervoltages
    TRAFO_B - two short feeders without undervoltages

Measurements are obtained by running a power flow for every 10 minute interval with
random household load profiles, so voltages and powers are consistent with the network.

Usage:
    python scripts/generate_demo_data.py [--days 42] [--out data/demo]
"""

import argparse
import os

import numpy as np
import pandapower as pp
import pandapower.topology as top
import pandas as pd
from pandapower.toolbox import select_subnet

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAFO_LV = 0.42  # transformer secondary voltage used for the "measurements"
SMM_START = {"TRAFO_A": 1000, "TRAFO_B": 2000}

# (main line type, segment length in km, number of segments) for every feeder
FEEDERS = {
    "TRAFO_A": [("NAYY 4x150 SE", 0.06, 6), ("NAYY 4x50 SE", 0.10, 9), ("NAYY 4x120 SE", 0.05, 5)],
    "TRAFO_B": [("NAYY 4x150 SE", 0.04, 5), ("NAYY 4x150 SE", 0.05, 4)],
}


def create_trafo_area(net, trafo_name, feeders, rng):
    """Adds external grid, MV/LV transformer and radial LV feeders to net.
    Every load gets a smart meter id (smm) and a number of phases."""
    hv_bus = pp.create_bus(net, vn_kv=20.0, name=f"{trafo_name} MV")
    pp.create_ext_grid(net, hv_bus, vm_pu=1.0)
    lv_bus = pp.create_bus(net, vn_kv=0.4, name=f"{trafo_name} LV")
    net.bus.loc[lv_bus, "aclass_id"] = "TR"
    pp.create_transformer(net, hv_bus, lv_bus, "0.4 MVA 20/0.4 kV", name=trafo_name)
    smm = SMM_START[trafo_name]
    for f, (std_type, length, n_segments) in enumerate(feeders, start=1):
        prev_bus = lv_bus
        for s in range(n_segments):
            bus = pp.create_bus(net, vn_kv=0.4, name=f"{trafo_name} F{f} B{s}")
            # only the first line of a feeder carries the feeder tag
            line_name = f"FEEDER-{f} {trafo_name}" if s == 0 else f"{trafo_name} F{f} L{s}"
            pp.create_line(net, prev_bus, bus, length, std_type, name=line_name)
            # one or two house connections at every main bus
            for h in range(rng.integers(1, 3)):
                house = pp.create_bus(net, vn_kv=0.4, name=f"{trafo_name} F{f} B{s} H{h}")
                pp.create_line(
                    net,
                    bus,
                    house,
                    rng.uniform(0.01, 0.04),
                    "NAYY 4x50 SE",
                    name=f"{trafo_name} F{f} S{s}{h}",
                )
                load = pp.create_load(net, house, p_mw=0.0, name=f"{trafo_name} house {smm}")
                net.load.loc[load, "smm"] = smm
                net.load.loc[load, "phases_true"] = 1 if rng.random() < 0.15 else 3
                smm += 1
            prev_bus = bus


def create_network(rng):
    net = pp.create_empty_network()
    net.bus["aclass_id"] = pd.Series(dtype=object)
    for trafo_name, feeders in FEEDERS.items():
        create_trafo_area(net, trafo_name, feeders, rng)
    net.load["smm"] = net.load["smm"].astype(int)
    return net


def household_profiles(index, n_loads, rng):
    """Random household active power profiles in kW, shape (len(index), n_loads)"""
    hours = index.hour.values + index.minute.values / 60
    morning = np.exp(-(((hours - 7.5) / 1.2) ** 2))
    evening = np.exp(-(((hours - 19.0) / 2.0) ** 2))
    # colder days have higher consumption
    day_factor = rng.lognormal(0.0, 0.25, size=index.normalize().nunique())
    day_idx = pd.factorize(index.normalize())[0]
    size = rng.uniform(0.6, 1.4, size=n_loads)
    base = 0.3 + 0.8 * morning[:, None] + 2.2 * evening[:, None]
    p = base * size[None, :] * day_factor[day_idx][:, None]
    p *= rng.lognormal(0.0, 0.35, size=p.shape)
    # heat pumps / EV charging: a few evenings with an extra 3-7 kW for 1-3 hours
    for j in range(n_loads):
        for day in rng.choice(day_idx.max() + 1, size=rng.integers(0, 6), replace=False):
            start = day * 144 + rng.integers(102, 126)  # 17:00 - 21:00
            p[start : start + rng.integers(6, 18), j] += rng.uniform(3.0, 7.0)
    return p


def simulate(net, index, rng):
    """Runs a power flow for every timestamp, returns voltages (pu) and powers (kW, kvar)"""
    n = len(net.load)
    p = household_profiles(index, n, rng)
    q = p * np.tan(np.arccos(rng.uniform(0.95, 0.99, size=n)))[None, :]
    vm = np.zeros_like(p)
    load_buses = net.load.bus.values
    net.trafo["vn_lv_kv"] = TRAFO_LV
    for t in range(len(index)):
        net.load["p_mw"] = p[t] / 1000
        net.load["q_mvar"] = q[t] / 1000
        pp.runpp(net, numba=False)
        vm[t] = net.res_bus.vm_pu.values[net.bus.index.get_indexer(load_buses)]
        if t % 1000 == 0:
            print(f"  power flow {t}/{len(index)}")
    net.trafo["vn_lv_kv"] = 0.4
    return vm, p, q


def to_measurements(net, index, vm, p, q, rng):
    """Converts simulation results to smart meter voltage and power tables"""
    voltage_rows, power_rows = [], []
    for j, load in enumerate(net.load.itertuples()):
        u = vm[:, j] * 230
        # constant phase unbalance and measurement noise
        phase_offset = rng.normal(0.0, 1.0, size=3)
        u_ph = u[:, None] + phase_offset[None, :] + rng.normal(0.0, 0.4, size=(len(u), 3))
        if load.phases_true == 1:
            u_ph[:, 1:] = np.nan
        # some missing measurements
        missing = rng.random(len(u)) < 0.01
        u_ph = u_ph[~missing]
        voltage_rows.append(
            pd.DataFrame(
                {
                    "date_time": index[~missing],
                    "smm": load.smm,
                    "u_1": u_ph[:, 0],
                    "u_2": u_ph[:, 1],
                    "u_3": u_ph[:, 2],
                    "u_123": np.nanmean(u_ph, axis=1),
                }
            )
        )
        power_rows.append(
            pd.DataFrame(
                {
                    "date_time": index,
                    "smm": load.smm,
                    "p": p[:, j],
                    "q": q[:, j],
                }
            )
        )
    voltage = pd.concat(voltage_rows).round(1)
    power = pd.concat(power_rows).round(3)
    return voltage, power


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--days", type=int, default=42)
    parser.add_argument("--start", default="2024-01-08")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=os.path.join(ROOT_DIR, "data", "demo"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    net = create_network(rng)
    index = pd.date_range(args.start, periods=args.days * 144, freq="10min")

    os.makedirs(args.out, exist_ok=True)
    for trafo_name in FEEDERS:
        print(f"Simulating {trafo_name}")
        trafo = net.trafo[net.trafo.name == trafo_name].index[0]
        loads = net.load[net.load.name.str.startswith(trafo_name)]
        # simulate only the area of this transformer
        buses = top.connected_component(top.create_nxgraph(net), net.trafo.at[trafo, "lv_bus"])
        subnet = select_subnet(net, list(buses))
        subnet.load = subnet.load.loc[loads.index]
        vm, p, q = simulate(subnet, index, rng)
        voltage, power = to_measurements(subnet, index, vm, p, q, rng)
        folder = os.path.join(args.out, trafo_name)
        os.makedirs(folder, exist_ok=True)
        voltage.to_csv(os.path.join(folder, "voltage.csv.gz"), index=False)
        power.to_csv(os.path.join(folder, "power.csv.gz"), index=False)
        print(
            f"  min voltage {voltage[['u_1', 'u_2', 'u_3']].min().min():.1f} V, "
            f"samples below 207 V: {(voltage[['u_1', 'u_2', 'u_3']] < 207).any(axis=1).sum()}"
        )

    net.load = net.load.drop(columns="phases_true")
    pp.to_json(net, os.path.join(args.out, "network.json"))
    print(f"Demo data written to {args.out}")


if __name__ == "__main__":
    main()
