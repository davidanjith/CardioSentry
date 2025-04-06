import pandas as pd
import numpy as np
import wfdb
from scipy.signal import resample
from sklearn.model_selection import GroupShuffleSplit
from src_ptbxl.data.dataset import ECGDataset
from src_ptbxl.config import DATA_PATH, TARGET_SAMPLES
import os

def load_and_split_data():
    #print(os.getcwd())
    #print(f'{DATA_PATH}/ptbxl_database.csv')
    metadata = pd.read_csv(f'{DATA_PATH}/ptbxl_database.csv')
    #print(metadata.head())

    scp = pd.read_csv(f'{DATA_PATH}/scp_statements.csv')
    scp_code_col = scp.columns[0]
    mi_codes = scp[(scp['diagnostic_subclass'] == 'MI') |
                   (scp['description'].str.contains('infarct', case=False))][scp_code_col].tolist()
    ischemia_codes = scp[scp['diagnostic_subclass'].isin(['STTC', 'NST_', 'ISC_'])][scp_code_col].tolist()
    normal_code = 'NORM'
    print(f"MI codes: {mi_codes}")
    print(f"Ischemia codes: {ischemia_codes}")
    signals, labels, patient_ids = [], [], []
    for _, row in metadata.iterrows():
        scp_codes = eval(row['scp_codes'])
        label = 2 if any(code in mi_codes for code in scp_codes) else 1 if any(
            code in ischemia_codes for code in scp_codes) else 0 if normal_code in scp_codes else None
        if label is None:
            continue
        record = wfdb.rdrecord(f"{DATA_PATH}/{row['filename_hr']}")
        resampled = np.array([resample(lead, TARGET_SAMPLES) for lead in record.p_signal.T])
        signals.append(resampled)
        labels.append(label)
        patient_ids.append(row['patient_id'])
    signals, labels, patient_ids = map(np.array, (signals, labels, patient_ids))

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=4)
    train_idx, temp_idx = next(gss.split(signals, labels, groups=patient_ids))
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=4)
    val_idx, test_idx = next(gss_val.split(signals[temp_idx], labels[temp_idx], groups=patient_ids[temp_idx]))



    return (ECGDataset(signals[train_idx], labels[train_idx], augment=True),
            ECGDataset(signals[val_idx], labels[val_idx]),
            ECGDataset(signals[test_idx], labels[test_idx]))