import os
import tempfile
from datetime import timedelta
import numpy as np
import pandas as pd
from HinetPy import Client, win32
from obspy import read
from obspy.signal.filter import bandpass

class HinetDownloader:
    """
    A class for downloading and processing Hi-net waveform data and saving RMS and spectral features.

    This class:
    - Downloads continuous waveform data from Hi-net using HinetPy
    - Converts data to SAC format (U-component only)
    - Computes RSAM and spectral amplitude averages for defined frequency bands
    - Calculates DSAR (MF/HF ratio)
    - Appends results to a CSV file and applies forward-fill for missing values

    Attributes:
        username (str): Hi-net username.
        password (str): Hi-net password.
        network (str): Hi-net network code (default is "0101").
        stations (list): List of station codes to be selected.
        output_dir (str): Directory where results are saved.
        csv_filename (str): Name of the CSV file to save results.
        csv_path (str): Full path to the output CSV file.
        client (HinetPy.Client): HinetPy client instance for communication with the server.
    """

    def __init__(self, username, password, network="0101", stations=["N.HKSH"],
                 output_dir="rms_results", csv_filename="10min_rms.csv"):
        """
        Initialize the downloader with user credentials, station settings, and output paths.

        Args:
            username (str): Hi-net username.
            password (str): Hi-net password.
            network (str, optional): Hi-net network code. Defaults to "0101".
            stations (list, optional): List of station codes. Defaults to ["N.HKSH"].
            output_dir (str, optional): Directory to save output data. Defaults to "rms_results".
            csv_filename (str, optional): Output CSV filename. Defaults to "10min_rms.csv".
        """
        self.username = username
        self.password = password
        self.network = network
        self.stations = stations
        self.output_dir = output_dir
        self.csv_filename = csv_filename
        self.csv_path = os.path.join(output_dir, csv_filename)

        os.makedirs(output_dir, exist_ok=True)
        self.client = Client(username, password)
        self.client.select_stations(network, stations)

    def download_and_process(self, start_time, end_time, interval_minutes=10):
        """
        Download waveform data and compute RMS and spectral amplitude features.

        For each time interval:
        - Download continuous waveform data using HinetPy
        - Convert to SAC files (U-component only)
        - Compute RSAM (2–5 Hz), MF (4.5–8 Hz), HF (8–16 Hz)
        - Calculate DSAR as MF/HF ratio
        - Append results to a CSV file
        - Apply forward-fill to handle missing data

        Args:
            start_time (datetime): Start datetime for data retrieval.
            end_time (datetime): End datetime for data retrieval.
            interval_minutes (int, optional): Window size in minutes. Defaults to 10.

        Raises:
            Exception: Any exception encountered during processing is caught and printed.
        """
        current = start_time

        if os.path.exists(self.csv_path):
            df_done = pd.read_csv(self.csv_path)
            done_start_times = set(df_done["time"])
        else:
            done_start_times = set()

        while current < end_time:
            t_str = current.strftime("%Y%m%d%H%M")
            start_str = current.strftime("%Y-%m-%d %H:%M")

            if start_str in done_start_times:
                print(f"✅ Skipping already processed interval: {start_str}")
                current += timedelta(minutes=interval_minutes)
                continue

            print(f"\n=== Processing {t_str} ===")
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    data_file, ctable = self.client.get_continuous_waveform(
                        self.network,
                        t_str,
                        interval_minutes,
                        outdir=tmp_dir,
                        cleanup=True
                    )
                    print(f"Downloaded to temp: {os.path.basename(data_file)}, {os.path.basename(ctable)}")

                    win32.extract_sac(data_file, ctable, outdir=tmp_dir)
                    print("SAC conversion completed (in temp)")

                    sac_files = [
                        os.path.join(tmp_dir, f)
                        for f in os.listdir(tmp_dir)
                        if f.endswith(".SAC") and ".U" in f
                    ]
                    print(f"U-component SAC files: {len(sac_files)}")

                    rsam_list = []
                    mf_list = []
                    hf_list = []

                    for sac_path in sac_files:
                        st = read(sac_path)
                        for tr in st:
                            data = tr.data.astype(np.float64) * 1e-9
                            df = tr.stats.sampling_rate
                            rsam_data = np.abs(bandpass(data, 2.0, 5.0, df)) * 1e9
                            mf_data   = np.abs(bandpass(data, 4.5, 8.0, df)) * 1e9
                            hf_data   = np.abs(bandpass(data, 8.0, 16.0, df)) * 1e9

                            rsam_list.append(np.mean(rsam_data))
                            mf_list.append(np.mean(mf_data))
                            hf_list.append(np.mean(hf_data))

                    avg_rsam = np.mean(rsam_list) if rsam_list else float("nan")
                    avg_mf   = np.mean(mf_list)   if mf_list else float("nan")
                    avg_hf   = np.mean(hf_list)   if hf_list else float("nan")
                    avg_dsar = avg_mf / avg_hf    if avg_hf and not np.isnan(avg_hf) else float("nan")

                    df_row = pd.DataFrame([{
                        "time": current + timedelta(minutes=interval_minutes),
                        "rsam": avg_rsam,
                        "mf": avg_mf,
                        "hf": avg_hf,
                        "dsar": avg_dsar
                    }])
                    df_row.to_csv(
                        self.csv_path,
                        mode="a",
                        header=not os.path.exists(self.csv_path),
                        index=False
                    )
                    print(f"Results appended to CSV: {self.csv_path}")

            except Exception as e:
                print(f"❌ Error occurred: {e}")

            current += timedelta(minutes=interval_minutes)

        # ===== Forward fill missing intervals =====
        print("⏳ Applying forward fill...")
        df = pd.read_csv(self.csv_path, parse_dates=["time"])
        df = df.sort_values("time").reset_index(drop=True)
        df = df.ffill()
        df.to_csv(self.csv_path, index=False)
        print("✅ Forward fill applied to CSV")


from datetime import datetime

if __name__ == "__main__":
    downloader = HinetDownloader(
        username="your_username",
        password="your_password",
        output_dir="data",
        csv_filename="observation_data.dat"
        )

    start = datetime(2020, 1, 1, 12, 0)
    end   = datetime(2020, 1, 1, 13, 0)

    downloader.download_and_process(start, end)

