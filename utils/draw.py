import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROCESSED_DIR = os.path.expanduser("~/dataset/processed/")


def plot_correlation_histogram(df, title='Correlation Histogram', xlabel='Correlation Range',
                               ylabel='Frequency'):
    plt.figure(figsize=(10, 6))
    plt.hist(df['CORRELATION'], bins=20, range=(-0.4, 0.4), edgecolor='black', color='cornflowerblue', alpha=0.8)

    plt.xticks(np.arange(-0.4, 0.45, 0.05))
    plt.xlim(-0.4, 0.4)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.show()


def plot_cdf(data, ax, xlabel, ylabel, title):
    sorted_data = np.sort(data)
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data))
    ax.plot(sorted_data, yvals, label=ylabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid()


def plot_missing_rate_cdf(itemid_file_path, patient_file_path):
    df_itemid = pd.read_csv(itemid_file_path)
    df_patient = pd.read_csv(patient_file_path)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    plot_cdf(df_itemid['PATIENT_LEVEL'], axs[0, 0], 'Missing Rate', 'Cumulative Probability',
             '(a) Medical Signs - Patient Level')
    plot_cdf(df_itemid['RECORD_LEVEL'], axs[0, 1], 'Missing Rate', 'Cumulative Probability',
             '(b) Medical Signs - Record Level')
    plot_cdf(df_patient['ITEMID_LEVEL'], axs[1, 0], 'Missing Rate', 'Cumulative Probability',
             '(c) Patients - Medical Signs Level')
    plot_cdf(df_patient['RECORD_LEVEL'], axs[1, 1], 'Missing Rate', 'Cumulative Probability',
             '(d) Patients - Record Level')

    fig.tight_layout()
    plt.show()


def main():
    correlations_file_path = os.path.join(PROCESSED_DIR, "correlations.csv")
    missing_rates_itemid_file_path = os.path.join(PROCESSED_DIR, "rates_numerical/missing_rates_per_itemid_720.csv")
    missing_rates_patient_file_path = os.path.join(PROCESSED_DIR, "rates_numerical/missing_rates_per_patient_720.csv")

    df_corr = pd.read_csv(correlations_file_path)
    plot_correlation_histogram(df_corr)

    plot_missing_rate_cdf(missing_rates_itemid_file_path, missing_rates_patient_file_path)


if __name__ == "__main__":
    main()
