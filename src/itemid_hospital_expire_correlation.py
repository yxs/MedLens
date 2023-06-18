import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import constants

LOGS_PER_ITEMID_DIR = os.path.expanduser("~/dataset/processed/logs_per_ITEMID")
OUTPUT_FILE = os.path.expanduser("~/dataset/processed/correlations.csv")


def compute_correlation(file):
    try:
        df = pd.read_csv(file, usecols=['ITEMID', 'VALUENUM', 'HOSPITAL_EXPIRE_FLAG'],
                         dtype={'ITEMID': int, 'VALUENUM': float, 'HOSPITAL_EXPIRE_FLAG': int})

        current_itemid = df['ITEMID'].iloc[0]
        df['VALUENUM'] = df['VALUENUM'].replace([np.inf, -np.inf], np.nan)
        cleaned_df = df.dropna(subset=['VALUENUM', 'HOSPITAL_EXPIRE_FLAG'])

        if np.all(cleaned_df['VALUENUM'] == cleaned_df['VALUENUM'].iloc[0]) or np.all(
                cleaned_df['HOSPITAL_EXPIRE_FLAG'] == cleaned_df['HOSPITAL_EXPIRE_FLAG'].iloc[0]):
            return current_itemid, None

        corr_value, _ = pearsonr(cleaned_df['VALUENUM'], cleaned_df['HOSPITAL_EXPIRE_FLAG'])
        return current_itemid, corr_value
    except Exception as e:
        print(f"Error processing file: {file}\n, Error message: {e}.")
        return None, None


def compute_correlations_concurrently():
    results = {}
    top_numerical_itemids = constants.TOP_CHARTEVENTS_ITEMID_NUMERICAL + constants.TOP_LABEVENTS_ITEMID

    files = [os.path.join(LOGS_PER_ITEMID_DIR, f) for id_ in top_numerical_itemids for f in
             os.listdir(LOGS_PER_ITEMID_DIR)
             if f.endswith('.csv') and int(f.split('_')[1][:-4]) == id_]

    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(compute_correlation, file): file for file in files}

        for future in tqdm(as_completed(futures), total=len(futures)):
            current_itemid, corr = future.result()
            if current_itemid is not None and corr is not None:
                results[current_itemid] = corr

    return results


if __name__ == "__main__":
    correlations = compute_correlations_concurrently()
    df = pd.DataFrame({'ITEMID': list(correlations.keys()), 'CORRELATION': list(correlations.values())})
    df = df.sort_values(by='CORRELATION', key=lambda x: x.abs(), ascending=False)
    df.to_csv(OUTPUT_FILE, index=False)
    print(df)
