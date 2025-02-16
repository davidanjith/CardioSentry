import os
import pandas as pd
import pickle
import numpy as np
from scipy.interpolate import interp1d

# File Paths
data_dir = r"C:\Users\David\PycharmProjects\CardioSentry\data_processing\PPG_DaLiA\PPG_FieldStudy"
output_file = "processed_data/ppg_dalia_cleaned/combined_ppg_dalia.parquet"
cached_raw_file = "processed_data/ppg_dalia_cleaned/raw_combined_ppg_dalia.parquet"

# Check if output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

all_subjects_data = []  # List to store interpolated data for all subjects
raw_all_subjects_data = []  # List to store raw (before interpolation) data

# Loop through all subject files (S1.pkl to S15.pkl)
for subject_id in range(1, 16):  # 15 subjects
    S_dir = os.path.join(data_dir, "S{0}".format(str(subject_id)))
    file_path = os.path.join(S_dir, f"S{subject_id}.pkl")

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File {file_path} not found, skipping...")
        continue

    print(f"Processing {file_path}...")

    # Load the pickle file
    with open(file_path, "rb") as file:
        data = pickle.load(file, encoding="latin1")

    # Extract required signals (keys in dict data)
    hr_labels = data["label"]  # Ground Truth HR values
    target_length = len(hr_labels)

    ecg_signal = data["signal"]["chest"]["ECG"].flatten()  # ECG Signal
    ppg_signal = data["signal"]["wrist"]["BVP"].flatten()  # PPG Signal
    rpeaks = data["rpeaks"]  # R-peak locations (for HRV)
    activity_labels = data["activity"].flatten()  # Activity data

    subject_id_array = np.full(len(hr_labels), subject_id)  # Repeat subject ID

    # Store raw (non-interpolated) data
    raw_df = pd.DataFrame({
        "ECG": ecg_signal[:target_length],
        "PPG": ppg_signal[:target_length],
        "HR": hr_labels,
        "Activity": activity_labels[:target_length],
        "Subject": subject_id_array
    })
    raw_all_subjects_data.append(raw_df)

    # Create time indices for each signal
    ecg_time = np.linspace(0, 1, len(ecg_signal))
    ppg_time = np.linspace(0, 1, len(ppg_signal))
    activity_time = np.linspace(0, 1, len(activity_labels))
    hr_time = np.linspace(0, 1, target_length)  # HR is the target length

    # Interpolate to match HR time axis
    interp_ecg = interp1d(ecg_time, ecg_signal, kind="linear", fill_value="extrapolate")
    interp_ppg = interp1d(ppg_time, ppg_signal, kind="linear", fill_value="extrapolate")
    interp_activity = interp1d(activity_time, activity_labels, kind="linear", fill_value="extrapolate")

    ecg_resampled = interp_ecg(hr_time)
    ppg_resampled = interp_ppg(hr_time)
    activity_resampled = interp_activity(hr_time)

    # Convert to DataFrame
    df = pd.DataFrame({
        "ECG": ecg_resampled,
        "PPG": ppg_resampled,
        "HR": hr_labels,
        "Activity": activity_resampled,
        "Subject": subject_id_array
    })

    all_subjects_data.append(df)

# Combine all subject data into a single DataFrame
if all_subjects_data:
    combined_df = pd.concat(all_subjects_data, ignore_index=True)
    combined_df.to_parquet(output_file, index=False)
    print(f"Saved combined interpolated data for all subjects to {output_file}")
else:
    print("No subjects processed successfully.")

# Save raw (non-interpolated) data
if raw_all_subjects_data:
    raw_combined_df = pd.concat(raw_all_subjects_data, ignore_index=True)
    raw_combined_df.to_parquet(cached_raw_file, index=False)
    print(f"Saved raw (non-interpolated) data for all subjects to {cached_raw_file}")

