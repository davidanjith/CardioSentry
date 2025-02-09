import os
import pandas as pd
import pickle

# File Paths
data_dir = "data_processing/PPG_DaLiA/PPG_FieldStudy"
output_dir = "/processed"

# Check if output directory already exists
os.makedirs(output_dir, exist_ok=True)

# Loop through all subject files (S1.pkl to S15.pkl)
for subject_id in range(1, 16):  # 15 subjects
    S_dir = os.path.join(data_dir, "S{0}".format(str(subject_id)))

    file_path = os.path.join(S_dir, f"S{subject_id}.pkl")
    file_path_corrected=file_path.replace("\\",'/')

    # Check if the  file exists..
    if not os.path.exists(file_path):
        print(f"File {file_path_corrected} not found, skipping...")
        continue

    print(f"Processing {file_path_corrected}...")

    # Load the pickle file
    with open(file_path_corrected, "rb") as file:
        data = pickle.load(file)

    # Extract required signals
    ppg_signal = data["signal"]["wrist"]["BVP"]  # PPG Signal
    ecg_signal = data["signal"]["chest"]["ECG"]  # ECG Signal
    hr_labels = data["label"]["HR"]  # Ground Truth Heart Rate
    activity_labels = data["label"]["activity"]  # Activity Labels

    # Convert to DataFrame
    df = pd.DataFrame({
        "PPG": ppg_signal.flatten(),
        "ECG": ecg_signal.flatten(),
        "HR": hr_labels,
        "Activity": activity_labels
    })

    # Each subjects cleaned data are stored separately
    output_file = os.path.join(output_dir, f"cleaned_S{subject_id}.parquet")
    df.to_parquet(output_file, index=False)
    print(f"Saved cleaned data for Subject {subject_id} to {output_file}")

print("All subjects processed successfully!")
