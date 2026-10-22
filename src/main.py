import argparse
import time
import warnings

import config
from data_loader import DataLoader
from pipeline import process_trafos

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate battery size and location needed to fix undervoltages in LV feeders."
    )
    parser.add_argument(
        "--network", default=config.NETWORK_PATH, help="pandapower network in JSON format"
    )
    parser.add_argument(
        "--data-dir",
        default=config.DATA_DIR,
        help="folder with one sub-folder of measurements per transformer",
    )
    parser.add_argument(
        "--trafo",
        action="append",
        help="transformer to process, can be repeated (default: all candidates found in data dir)",
    )
    parser.add_argument(
        "--all-feeders",
        action="store_true",
        help="save results for all feeders, not only the ones that need a battery",
    )
    parser.add_argument("--out", default="battery_results.csv", help="output CSV file")
    return parser.parse_args()


def main():
    args = parse_args()
    trafos_list = args.trafo
    if not trafos_list:
        trafos_list = DataLoader(data_dir=args.data_dir).find_trafo_candidates()
        print("Transformer candidates:", trafos_list)

    time0 = time.time()
    res = process_trafos(trafos_list, args.network, args.data_dir, args.all_feeders)
    print(res.to_string())
    res.to_csv(args.out, index=False)
    print(f"Results saved to {args.out}, processing took {time.time() - time0:.1f} s")


if __name__ == "__main__":
    main()
