import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import logging

import constants

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

LOG_DIR = os.path.expanduser("~/dataset/processed/logs_per_SUBJECT_ID_DISCHTIME")
TIME_SERIES_BASE_DIR = os.path.expanduser("~/dataset/processed/")
ITEMIDS = constants.TOP_CHARTEVENTS_ITEMID + constants.TOP_LABEVENTS_ITEMID


def generate_time_series_file(file_name, hours, time_series_base_dir):
    subject_id = int(file_name.split("_")[1])
    log_file = os.path.join(LOG_DIR, file_name)

    if not os.path.exists(log_file):
        return

    try:
        log_data = pd.read_csv(log_file, usecols=["SUBJECT_ID", "ITEMID", "CHARTTIME", "VALUENUM", "DISCHTIME",
                                                  "HOSPITAL_EXPIRE_FLAG"], parse_dates=["CHARTTIME", "DISCHTIME"],
                               dtype={"SUBJECT_ID": int, "ITEMID": int, "VALUENUM": float})
    except ValueError as e:
        logging.error(f"Error reading {log_file}: {e}")
        return

    if log_data.empty:
        return file_name

    dischtime = log_data.iloc[0]["DISCHTIME"]
    hospital_expire_flag = log_data.iloc[0]["HOSPITAL_EXPIRE_FLAG"]
    dischtime = dischtime.ceil("1H")

    for hour in hours:
        start_time = dischtime - pd.Timedelta(hours=hour)
        time_index = pd.date_range(start_time, dischtime, freq="H", inclusive="right").sort_values(ascending=False)

        item_data_frames = []
        grouped_data = log_data.groupby('ITEMID')
        for itemid in ITEMIDS:
            item_data = grouped_data.get_group(itemid) if itemid in grouped_data.groups else None
            if item_data is not None:
                item_data = item_data.set_index(pd.DatetimeIndex(item_data["CHARTTIME"]))["VALUENUM"]
                item_data_avg = item_data.resample("H", closed='right', label='right').mean()
            else:
                item_data_avg = pd.Series(index=time_index, dtype=float)
            item_data_frames.append(item_data_avg.to_frame(itemid))

        time_series = pd.concat(item_data_frames, axis=1, join='outer', sort=False)
        time_series = time_series.reindex(time_index)

        time_series["SUBJECT_ID"] = subject_id
        time_series["HOSPITAL_EXPIRE_FLAG"] = hospital_expire_flag

        time_series.reset_index(inplace=True)
        time_series.rename(columns={'index': 'TIME'}, inplace=True)
        time_series.insert(0, 'TIME\ITEMID', time_series['TIME'])
        time_series.drop(columns=['TIME'], inplace=True)
        time_series.set_index('TIME\ITEMID', inplace=True)

        time_series_dir = os.path.join(time_series_base_dir, f"time_series_{hour}")
        os.makedirs(time_series_dir, exist_ok=True)
        time_series_file = os.path.join(time_series_dir, f"{file_name.replace('.csv', '')}_time-series-{hour}.csv")
        time_series.to_csv(time_series_file)


def main():
    mode = "run"

    hours = [1 * 24, 2 * 24, 7 * 24, 30 * 24]
    file_names = []
    if mode == "run":
        file_names = [f for f in os.listdir(LOG_DIR) if f.startswith("log_")]
    elif mode == "test":
        file_names = ["log_31_2108-08-30_15-00-00.csv",
                      "log_23_2153-09-08_19-10-00.csv",
                      "log_23_2157-10-25_14-00-00.csv",
                      "log_27_2191-12-03_14-45-00.csv",
                      "log_2_2138-07-21_15-48-00.csv"]

    with ThreadPoolExecutor(max_workers=12) as executor:
        results = list(tqdm(executor.map(
            lambda f: generate_time_series_file(f, hours, TIME_SERIES_BASE_DIR),
            file_names),
            total=len(file_names)))
    empty_log_files = [filename for filename in results if filename is not None]
    if empty_log_files:
        logging.info(f"Empty log files: {', '.join(empty_log_files)}")
        with open(os.path.join(TIME_SERIES_BASE_DIR, 'empty_log_files.txt'), 'w') as f:
            f.write('\n'.join(empty_log_files))


if __name__ == "__main__":
    main()
