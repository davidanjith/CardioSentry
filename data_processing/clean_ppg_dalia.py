import os
import pandas as pd
import pickle
import numpy as np
from scipy.interpolate import interp1d

# File Paths
data_dir = r"C:\Users\David\PycharmProjects\CardioSentry\data_processing\PPG_DaLiA\PPG_FieldStudy"
output_dir = "processed_data/ppg_dalia_cleaned"

# Check if output directory already exists
os.makedirs(output_dir, exist_ok=True)

# Loop through all subject files (S1.pkl to S15.pkl)
for subject_id in range(1, 16):  # 15 subjects
    S_dir = os.path.join(data_dir, "S{0}".format(str(subject_id)))

    file_path = os.path.join(S_dir, f"S{subject_id}.pkl")

    # file_path_corrected=file_path.replace("\\",'/')

    # Check if the  file exists..
    if not os.path.exists(file_path):
        print(f"File {file_path} not found, skipping...")
        continue

    print(f"Processing {file_path}...")

    # Load the pickle file
    with open(file_path, "rb") as file:
        data = pickle.load(file, encoding="latin1")
        # print(type(data))
        # print(data.keys())
        # print(data["subject"])  # Check its type
        # print(data["label"])  # Print a sample of the hr data

    # Extract required signals (keys in dict data)
    hr_labels = data["label"]  # Ground Truth HR values
    target_length = len(hr_labels)
    #Resample and keep alignment
    ecg_signal = data["signal"]["chest"]["ECG"]  # ECG Signal
    ecg_signal = resample(ecg_signal, target_length)

    ppg_signal = data["signal"]["wrist"]["BVP"]  # PPG Signal
    ppg_signal = resample(ppg_signal, target_length)

    rpeaks = data["rpeaks"]  # R-peak locations (for HRV)
    activity_labels = data["activity"]  # Activity data
    activity_labels = resample(activity_labels, target_length)
    subject_id = data["subject"]  # Subject ID

    #Check the dims of the numpy arrays
    # print("ECG Shape:", np.shape(ecg_signal)) #needs flatenning and downsampling
    # print("PPG Shape:", np.shape(ppg_signal)) #needs flatenning and downsampling
    # print("R-peaks Shape:", np.shape(rpeaks))
    # print("HR Shape:", np.shape(hr_labels))
    # print("Activity Shape:", np.shape(activity_labels)) #needs flattening and alignment
    # print("Subject Shape:", np.shape(subject_id)) #needs repetition


    # Convert to DataFrame
    df = pd.DataFrame({
        "ECG": ecg_signal.flatten(),
        "PPG": ppg_signal.flatten(),
        "HR": hr_labels,
        "Activity": activity_labels.flatten(),
        "Subject": subject_id
    })

    # Each subjects cleaned data are stored separately
    output_file = os.path.join(output_dir, f"cleaned_S{subject_id}.parquet")
    df.to_parquet(output_file, index=False)
    print(f"Saved cleaned data for Subject {subject_id} to {output_file}")

print("All subjects processed successfully!")
