import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import constants

DATA_DIR = os.path.expanduser("~/dataset/mimic-iii-1.4")
STATS_OUTPUT_DIR = os.path.expanduser("~/dataset/processed/stats")

if not os.path.exists(STATS_OUTPUT_DIR):
    os.makedirs(STATS_OUTPUT_DIR)


def calculate_stats(chunk, itemids, stats_results):
    for itemid in itemids:
        filtered_df = chunk[chunk["ITEMID"] == itemid]
        if filtered_df["VALUENUM"].count() == 0:
            continue
        if not stats_results[itemid]:  # Initialize the dictionary if not already initialized
            stats_results[itemid] = {"count": 0, "sum": 0, "MIN": np.inf, "MAX": -np.inf}
        stats_results[itemid]["count"] += filtered_df["VALUENUM"].count()
        stats_results[itemid]["sum"] += filtered_df["VALUENUM"].sum()
        if stats_results[itemid]["count"] > 0:
            stats_results[itemid]["MEAN"] = stats_results[itemid]["sum"] / stats_results[itemid]["count"]
        stats_results[itemid]["MIN"] = min(stats_results[itemid]["MIN"], filtered_df["VALUENUM"].min())
        stats_results[itemid]["MAX"] = max(stats_results[itemid]["MAX"], filtered_df["VALUENUM"].max())


def process_file(file_path, itemids, chunksize=10 ** 6):
    stats_results = {itemid: {} for itemid in itemids}

    with ThreadPoolExecutor(max_workers=12) as executor:
        for chunk in tqdm(pd.read_csv(file_path, usecols=["ITEMID", "VALUENUM"], dtype={"VALUENUM": "float64"},
                                      chunksize=chunksize), unit="chunk"):
            executor.submit(calculate_stats, chunk, itemids, stats_results)

    # Calculate the mean from the count and sum
    for itemid in stats_results:
        stats_results[itemid]["MEAN"] = stats_results[itemid]["sum"] / stats_results[itemid]["count"]
        del stats_results[itemid]["count"]
        del stats_results[itemid]["sum"]

    return stats_results


def calc_mean_max_min():
    chartevents_file = os.path.join(DATA_DIR, "CHARTEVENTS.csv")
    labevents_file = os.path.join(DATA_DIR, "LABEVENTS.csv")

    chartevents_stats = process_file(chartevents_file, constants.TOP_CHARTEVENTS_ITEMID_NUMERICAL)
    labevents_stats = process_file(labevents_file, constants.TOP_LABEVENTS_ITEMID)

    stats_results = {**chartevents_stats, **labevents_stats}

    stats_df = pd.DataFrame.from_dict(stats_results, orient='index',
                                      columns=["MEAN", "MAX", "MIN"]).reset_index().rename(columns={'index': 'ITEMID'})

    stats_df.to_csv(os.path.join(STATS_OUTPUT_DIR, "ITEMID_valuenum_mean-max-min.csv"), index=False)


def main():
    calc_mean_max_min()


if __name__ == "__main__":
    main()
