import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import constants

DATA_DIR = os.path.expanduser("~/dataset/mimic-iii-1.4")
LOGS_PER_SUBJECT_ID_DISCHTIME_DIR = os.path.expanduser("~/dataset/processed/logs_per_SUBJECT_ID_DISCHTIME")
LOGS_PER_ITEMID_DIR = os.path.expanduser("~/dataset/processed/logs_per_ITEMID")


def get_expire_flags_and_sorted_dischtimes(admissions_file):
    admissions_df = pd.read_csv(admissions_file, usecols=['SUBJECT_ID', 'DISCHTIME', 'HOSPITAL_EXPIRE_FLAG'],
                                dtype={'SUBJECT_ID': 'int', 'HOSPITAL_EXPIRE_FLAG': 'int'}, parse_dates=['DISCHTIME'])
    expire_flags = {}

    for _, row in admissions_df.iterrows():
        subject_id, dischtime, expire_flag = row['SUBJECT_ID'], row['DISCHTIME'], row['HOSPITAL_EXPIRE_FLAG']
        dischtime_timestamp = int(dischtime.timestamp())
        if subject_id not in expire_flags:
            expire_flags[subject_id] = {}
        expire_flags[subject_id][dischtime_timestamp] = expire_flag

    for subject_id, dischtime_expire_flags in expire_flags.items():
        sorted_dischtime_expire_flags = dict(sorted(dischtime_expire_flags.items()))
        expire_flags[subject_id] = sorted_dischtime_expire_flags

    return expire_flags


def create_header_file(file_path, header):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write(header)


def create_header_files(expire_flags, chartevents_itemids, labevents_itemids):
    header = "SUBJECT_ID,ITEMID,CHARTTIME,VALUE,VALUENUM,DISCHTIME,HOSPITAL_EXPIRE_FLAG\n"

    with ThreadPoolExecutor(max_workers=12) as executor:
        for subject_id, dischtime_expire_flags in expire_flags.items():
            for dischtime_timestamp in dischtime_expire_flags.keys():
                dischtime = pd.to_datetime(dischtime_timestamp, unit='s')
                dischtime_str = dischtime.strftime('%Y-%m-%d_%H-%M-%S')
                per_charge_dir = os.path.join(LOGS_PER_SUBJECT_ID_DISCHTIME_DIR,
                                              f"log_{subject_id}_{dischtime_str}.csv")
                executor.submit(create_header_file, per_charge_dir, header)

        for itemid in chartevents_itemids + labevents_itemids:
            per_itemid_dir = os.path.join(LOGS_PER_ITEMID_DIR, f"log_{itemid}.csv")
            executor.submit(create_header_file, per_itemid_dir, header)


def process_chunk(chunk, itemids, expire_flags):
    filtered_chunk = chunk[chunk['ITEMID'].isin(itemids)]

    for _, group in filtered_chunk.groupby('SUBJECT_ID'):
        subject_id = group['SUBJECT_ID'].iloc[0]
        subject_expire_flags = expire_flags[subject_id]
        dischtime_timestamps = list(subject_expire_flags.keys())

        for idx, current_dischtime_timestamp in enumerate(dischtime_timestamps):
            current_dischtime = pd.to_datetime(current_dischtime_timestamp, unit='s')
            current_dischtime_str = current_dischtime.strftime('%Y-%m-%d_%H-%M-%S')
            per_charge_dir = os.path.join(LOGS_PER_SUBJECT_ID_DISCHTIME_DIR,
                                          f"log_{subject_id}_{current_dischtime_str}.csv")
            if idx == 0:
                prev_dischtime = pd.Timestamp.min
            else:
                prev_dischtime = pd.to_datetime(dischtime_timestamps[idx - 1], unit='s')
            within_dischtime = (group['CHARTTIME'] > prev_dischtime) & (group['CHARTTIME'] <= current_dischtime)
            subject_data_within_dischtime = group[within_dischtime].copy()
            subject_data_within_dischtime['DISCHTIME'] = current_dischtime
            subject_data_within_dischtime['HOSPITAL_EXPIRE_FLAG'] = subject_expire_flags[current_dischtime_timestamp]
            subject_data_within_dischtime.to_csv(per_charge_dir, index=False, mode='a', header=False)
            for itemid, itemid_data in subject_data_within_dischtime.groupby('ITEMID'):
                per_itemid_dir = os.path.join(LOGS_PER_ITEMID_DIR, f"log_{itemid}.csv")
                itemid_data.to_csv(per_itemid_dir, index=False, mode='a', header=False)

    return True


def process_file(file_path, itemids, expire_flags):
    chunksize = 10 ** 6

    total_chunks = sum(
        1 for _ in pd.read_csv(file_path, usecols=['SUBJECT_ID'], dtype={'SUBJECT_ID': 'int'}, chunksize=chunksize))

    with ThreadPoolExecutor(max_workers=12) as executor:
        for chunk in tqdm(pd.read_csv(file_path, usecols=['SUBJECT_ID', 'ITEMID', 'CHARTTIME', 'VALUE', 'VALUENUM'],
                                      dtype={'SUBJECT_ID': int, 'ITEMID': int, 'VALUE': str, 'VALUENUM': float},
                                      parse_dates=['CHARTTIME'], chunksize=chunksize),
                          total=total_chunks, desc=f"Processing {os.path.basename(file_path)}"):
            executor.submit(process_chunk, chunk, itemids, expire_flags)


def main():
    admissions_file = os.path.join(DATA_DIR, "ADMISSIONS.csv")
    chartevents_file = os.path.join(DATA_DIR, "CHARTEVENTS.csv")
    labevents_file = os.path.join(DATA_DIR, "LABEVENTS.csv")

    expire_flags = get_expire_flags_and_sorted_dischtimes(admissions_file)

    if not os.path.exists(LOGS_PER_SUBJECT_ID_DISCHTIME_DIR):
        os.makedirs(LOGS_PER_SUBJECT_ID_DISCHTIME_DIR)

    if not os.path.exists(LOGS_PER_ITEMID_DIR):
        os.makedirs(LOGS_PER_ITEMID_DIR)

    create_header_files(expire_flags, constants.HARUTYUNYAN_2019_MULTITASK_ITEMID_CHARTEVENTS,
                        constants.HARUTYUNYAN_2019_MULTITASK_ITEMID_LABEVENTS)

    process_file(chartevents_file, constants.HARUTYUNYAN_2019_MULTITASK_ITEMID_CHARTEVENTS, expire_flags)
    process_file(labevents_file, constants.HARUTYUNYAN_2019_MULTITASK_ITEMID_LABEVENTS, expire_flags)


if __name__ == "__main__":
    main()
