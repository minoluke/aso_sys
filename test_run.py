#!/usr/bin/env python3

import os
import sys
import json
import warnings
import numpy as np
from datetime import timedelta, datetime

# Suppress warnings and NumPy floating-point errors
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

# Import project modules
from modules import *

# === Parameters (edit as needed) ===
observation_data_key = "tremor"
look_backward = 60.0  # days
look_forward = 30.0   # days
cv_index = 0
start_period = "2010-01-01"
end_period = "2022-12-31"

# Load data stream definitions from JSON
with open(os.path.join("data", "data_streams.json"), "r", encoding="utf-8") as f:
    data_streams_dict = json.load(f)


def quick_download():
    """Download a short Hi-net waveform segment and compute seismic features.

    This function:
        - Connects to the Hi-net server using user credentials
        - Downloads 10 minutes of waveform data for testing
        - Extracts SAC files and computes RSAM, MF, HF, and DSAR features
        - Appends the results to a CSV file in the `data/` directory

    Note:
        Replace 'your_username' and 'your_password' with valid Hi-net credentials.
    """
    print("🚚 Running quick Hi-net data download...")

    downloader = HinetDownloader(
        username="your_username",
        password="your_password",
        output_dir="data",
        csv_filename="observation_data.dat"
    )

    start = datetime(2020, 1, 1, 12, 0)
    end = datetime(2020, 1, 1, 12, 10)

    downloader.download_and_process(start, end)

    print("✅ Download completed.")


def quick_train_and_test():
    """Train and test a single model run using fixed parameters.

    This function:
        - Loads eruption metadata and selects test period
        - Chooses data stream features based on the selected category (e.g., 'tremor')
        - Trains a model while excluding a 6-month eruption window
        - Tests the model on the same observation range

    Raises:
        KeyError: If the specified observation_data_key is not found in data_streams_dict.
    """
    print("🚀 Starting quick model train/test...")

    obs_data = ObservationData()
    eruption_time = obs_data.tes[cv_index]
    data_streams = data_streams_dict[observation_data_key]

    # Training
    trainer = TrainModel(
        ti=start_period,
        tf=end_period,
        look_backward=look_backward,
        look_forward=look_forward,
        data_streams=data_streams,
        od=observation_data_key,
        cv=cv_index
    )
    trainer.train(
        cv=cv_index,
        ti=start_period,
        tf=end_period,
        exclude_dates=[[eruption_time - timedelta(days=180), eruption_time + timedelta(days=180)]]
    )

    # Testing
    tester = TestModel(
        ti=start_period,
        tf=end_period,
        look_backward=look_backward,
        look_forward=look_forward,
        data_streams=data_streams,
        od=observation_data_key,
        cv=cv_index
    )
    tester.test(
        cv=cv_index,
        ti=start_period,
        tf=end_period
    )

    print("✅ Test run completed.")


if __name__ == "__main__":
    #quick_download()
    quick_train_and_test()
