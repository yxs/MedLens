import math
import os

from collections import defaultdict
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import constants

DATA_DIR = os.path.expanduser("~/dataset/mimic-iii-1.4")
PROCESSED_DIR = os.path.expanduser("~/dataset/processed")


def write_top_items_to_csv(top_items_dict):
    table_itemids = defaultdict(dict)
    for item_id, (table, count, value, valuenum) in top_items_dict.items():
        table_itemids[table][item_id] = (count, value, valuenum)

    for table, item_dict in table_itemids.items():
        num_items = len(item_dict)
        filename = f"top_{table}_{num_items}_itemids.csv"

        with open(os.path.join(PROCESSED_DIR, filename), "w") as f:
            f.write("ITEMID,COUNT,VALUE,VALUENUM\n")
            for item_id, (count, value, valuenum) in item_dict.items():
                f.write(f"{item_id},{count},{value},{valuenum}\n")


def custom_function(df):
    return df.groupby("ITEMID").first().reset_index()


def count_itemids(file_paths):
    itemid_dict = defaultdict(lambda: [None, 0, None, None])

    for path in file_paths:
        table = os.path.splitext(os.path.basename(path))[0].upper()
        ddf = dd.read_csv(path, usecols=["ITEMID", "VALUE", "VALUENUM"], blocksize='64MB',
                          dtype={'VALUE': 'object'})

        grouped = ddf.groupby("ITEMID").size().compute()
        first_values = ddf.map_partitions(custom_function).compute()

        for itemid, count in grouped.items():
            itemid_dict[itemid][0] = table
            itemid_dict[itemid][1] += count

        for idx, row in first_values.iterrows():
            itemid = row["ITEMID"]
            itemid_dict[itemid][2] = str(row["VALUE"])
            itemid_dict[itemid][3] = str(row["VALUENUM"])

    return itemid_dict


def is_numerical(valuenum):
    try:
        float_val = float(valuenum)
        return not math.isnan(float_val)
    except ValueError:
        return False


def calc_percentage(itemid_dict):
    total_count = sum([count for _, (_, count, _, _) in itemid_dict.items()])
    total_numerical_count = sum(
        [count for _, (_, count, _, valuenum) in itemid_dict.items() if is_numerical(valuenum)])
    top_itemids_total_count = sum(
        [itemid_dict[itemid][1] for itemid in constants.TOP_CHARTEVENTS_ITEMID + constants.TOP_LABEVENTS_ITEMID if
         itemid in itemid_dict])
    top_numerical_itemids_total_count = sum(
        [itemid_dict[itemid][1] for itemid in
         constants.TOP_CHARTEVENTS_ITEMID_NUMERICAL + constants.TOP_LABEVENTS_ITEMID if
         itemid in itemid_dict])
    percentage = (top_itemids_total_count / total_count) * 100
    numerical_percentage = (top_numerical_itemids_total_count / total_numerical_count) * 100
    print(f"Top ITEMIDs count: {top_itemids_total_count}")
    print(f"Top numerical ITEMIDs count: {top_numerical_itemids_total_count}\n")

    print(f"Unique ITEMIDs count: {len(itemid_dict)}")
    print(f"Total ITEMIDs count: {total_count}")
    print(f"Total numerical ITEMIDs count: {total_numerical_count}\n")

    print(f"Percentage of top ITEMIDs: {percentage}%")
    print(f"Percentage of top numerical ITEMIDs: {numerical_percentage}%")


def process_itemids():
    file_paths = [
        os.path.join(DATA_DIR, "CHARTEVENTS.csv"),
        os.path.join(DATA_DIR, "LABEVENTS.csv")
    ]

    itemid_dict = count_itemids(file_paths)

    calc_percentage(itemid_dict)

    # top_items = sorted(itemid_dict.items(), key=lambda x: x[1][1], reverse=True)[:constants.TOP_N]
    # top_items_dict = {}
    # for itemid, (table, count, value, valuenum) in top_items:
    #     top_items_dict[itemid] = [table, count, value, valuenum]
    #
    # write_top_items_to_csv(top_items_dict)


if __name__ == "__main__":
    with ProgressBar():
        process_itemids()
