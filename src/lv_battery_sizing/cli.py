import argparse
import logging
import time
import warnings

from . import config
from .data_loader import DataLoader
from .pipeline import process_trafos

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="lvbat",
        description="Calculate battery size and location that fix undervoltages in LV feeders.",
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
        help="transformer to process, can be repeated (default: all candidates in data dir)",
    )
    parser.add_argument(
        "--all-feeders",
        action="store_true",
        help="save results for all feeders, not only the ones that need a battery",
    )
    parser.add_argument("--out", default="battery_results.csv", help="output CSV file")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true", help="show debug messages")
    verbosity.add_argument("-q", "--quiet", action="store_true", help="show only warnings")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    # third party libraries are noisy about deprecations
    warnings.filterwarnings("ignore")

    trafos_list = args.trafo
    if not trafos_list:
        trafos_list = DataLoader(data_dir=args.data_dir).find_trafo_candidates()
        logger.info("Transformer candidates: %s", trafos_list)

    time0 = time.time()
    res = process_trafos(trafos_list, args.network, args.data_dir, args.all_feeders)
    print(res.to_string())
    res.to_csv(args.out, index=False)
    logger.info("Results saved to %s, processing took %.1f s", args.out, time.time() - time0)


if __name__ == "__main__":
    main()
