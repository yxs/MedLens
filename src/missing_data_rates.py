import os
import traceback

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import constants
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TIME_SERIES_DIR_TEMPLATE = os.path.expanduser("~/dataset/processed/time_series_{}/")
OUTPUT_DIR = os.path.expanduser("~/dataset/processed/rates_numerical")

ITEMIDS = constants.TOP_CHARTEVENTS_ITEMID_NUMERICAL + constants.TOP_LABEVENTS_ITEMID
ITEMID_COUNT = len(ITEMIDS)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def save_missing_rates_to_csv(missing_rates, duration):
    itemid_data = pd.DataFrame(columns=['ITEMID', 'PATIENT_LEVEL', 'RECORD_LEVEL'])
    patient_data = pd.DataFrame(columns=['SUBJECT_ID', 'ITEMID_LEVEL', 'RECORD_LEVEL'])

    for key in ITEMIDS:
        itemid_data = pd.concat([itemid_data, pd.DataFrame([{
            'ITEMID': key,
            'PATIENT_LEVEL': missing_rates['patient_level_per_itemid'][key],
            'RECORD_LEVEL': missing_rates['record_level_per_itemid'][key]
        }])], ignore_index=True)

    for key in missing_rates['itemid_level_per_patient']:
        patient_data = pd.concat([patient_data, pd.DataFrame([{
            'SUBJECT_ID': key,
            'ITEMID_LEVEL': missing_rates['itemid_level_per_patient'][key],
            'RECORD_LEVEL': missing_rates['record_level_per_patient'][key]
        }])], ignore_index=True)

    itemid_data.to_csv(os.path.join(OUTPUT_DIR, f"missing_rates_per_itemid_{duration}.csv"), index=False)
    patient_data.to_csv(os.path.join(OUTPUT_DIR, f"missing_rates_per_patient_{duration}.csv"), index=False)


def calc_missing_rates(stats, duration, patient_count):
    missing_rates = {
        'patient_level_per_itemid': {key: 0 for key in ITEMIDS},
        'record_level_per_itemid': {key: 0 for key in ITEMIDS},
        'itemid_level_per_patient': {},
        'record_level_per_patient': {}
    }

    for key in ITEMIDS:
        missing_rates['patient_level_per_itemid'][key] = 1 - (stats['patient_level_per_itemid'][key] / patient_count)
        missing_rates['record_level_per_itemid'][key] = 1 - (
                stats['record_level_per_itemid'][key] / (patient_count * duration))

    for key in stats['itemid_level_per_patient']:
        missing_rates['itemid_level_per_patient'][key] = 1 - (stats['itemid_level_per_patient'][key] / ITEMID_COUNT)

    for key in stats['record_level_per_patient']:
        missing_rates['record_level_per_patient'][key] = 1 - (
                stats['record_level_per_patient'][key] / (ITEMID_COUNT * duration))

    return missing_rates


def missing_rates_stats(time_series_dir, filename):
    subject_id = None
    item_count = {}
    try:
        subject_id, item_count = process_time_series_file(time_series_dir, filename)
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error processing file {filename}: {e}\n{tb}")
    stats = {
        'patient_level_per_itemid': {key: 0 for key in ITEMIDS},
        'record_level_per_itemid': {key: 0 for key in ITEMIDS},
        'itemid_level_per_patient': {},
        'record_level_per_patient': {}
    }

    for key in item_count:
        if key in ITEMIDS:
            stats['patient_level_per_itemid'][key] += 1
            stats['record_level_per_itemid'][key] += item_count[key]

    stats['itemid_level_per_patient'][subject_id] = len(item_count)
    stats['record_level_per_patient'][subject_id] = sum(item_count.values())
    return stats


def process_time_series_file(time_series_dir, filename):
    time_series_file = os.path.join(time_series_dir, filename)
    subject_id = get_subject_id_from_filename(filename)

    time_series_data = pd.read_csv(time_series_file)
    time_series_data = time_series_data.dropna(how='all')

    itemids_columns = [col for col in time_series_data.columns if col.isdigit()]
    time_series_data = time_series_data[itemids_columns]

    item_count = time_series_data.iloc[:, 1:].count().to_dict()
    # some ITEMID will be float while read.
    int_item_count = {int(float(key)): value for key, value in item_count.items()}  # k is ITEMID, v is count

    return subject_id, int_item_count


def get_patient_count(time_series_dir):
    return sum(1 for entry in os.scandir(time_series_dir) if entry.is_file())


def get_subject_id_from_filename(filename):
    return int(filename.split('_')[1])


def main():
    mode = "run"

    for duration in constants.DURATIONS:
        time_series_dir = TIME_SERIES_DIR_TEMPLATE.format(duration)
        patient_count = get_patient_count(time_series_dir)
        if mode == "run":
            stats = {
                # v: The number of occurrences of ITEMID in all csv
                'patient_level_per_itemid': {key: 0 for key in ITEMIDS},
                # v: The number of ITEMID records appearing in all csv
                'record_level_per_itemid': {key: 0 for key in ITEMIDS},
                # v: The number of occurrences of ITEMID in the csv of the corresponding patient
                'itemid_level_per_patient': {},
                # v: The total number of non-empty records for this patient
                'record_level_per_patient': {}
            }
            csv_files = [filename for filename in os.listdir(time_series_dir) if
                         filename.endswith(f"_time-series-{duration}.csv")]

            with ThreadPoolExecutor(max_workers=12) as executor:
                futures = {executor.submit(missing_rates_stats, time_series_dir, filename): filename for filename in
                           csv_files}

                for future in tqdm(as_completed(futures), total=len(csv_files), desc="Processing files"):
                    file_stats = future.result()
                    for key in file_stats:
                        if key in ['patient_level_per_itemid', 'record_level_per_itemid']:
                            for itemid in file_stats[key]:
                                stats[key][itemid] += file_stats[key][itemid]
                        else:
                            stats[key].update(file_stats[key])

            missing_rates = calc_missing_rates(stats, duration, patient_count)
            save_missing_rates_to_csv(missing_rates, duration)
        elif mode == "test":
            subject_id = 2
            csv_file = next((filename for filename in os.listdir(time_series_dir) if
                             filename.startswith(f"log_{subject_id}_") and filename.endswith(
                                 f"_time-series-{duration}.csv")), None)
            stats = {
                # v: The number of occurrences of ITEMID in all csv
                'patient_level_per_itemid': {key: 0 for key in ITEMIDS},
                # v: The number of ITEMID records appearing in all csv
                'record_level_per_itemid': {key: 0 for key in ITEMIDS},
                # v: The number of occurrences of ITEMID in the csv of the corresponding patient
                'itemid_level_per_patient': {},
                # v: The total number of non-empty records for this patient
                'record_level_per_patient': {}
            }
            file_stats = missing_rates_stats(time_series_dir, csv_file)
            for key in file_stats:
                if key in ['patient_level_per_itemid', 'record_level_per_itemid']:
                    for itemid in file_stats[key]:
                        stats[key][itemid] += file_stats[key][itemid]
                else:
                    stats[key].update(file_stats[key])

            missing_rates = calc_missing_rates(stats, duration, patient_count)
            save_missing_rates_to_csv(missing_rates, duration)


if __name__ == "__main__":
    main()
