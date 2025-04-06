from src_ptbxl.data.dataset import ECGDataset
from src_ptbxl.config import DATA_PATH, TARGET_SAMPLES
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample

def get_ecg_by_condition(condition='NORM', max_samples=1, augment=False):
    """
    Get ECGs matching specific SCP codes in ECGDataset format
    Args:
        condition: SCP code to filter by (e.g., 'NORM', 'AMI')
        max_samples: maximum number of samples to return
        augment: whether to apply augmentation (for training)
    Returns:
        ECGDataset containing matching samples
    """
    # Load metadata
    metadata = pd.read_csv(f'{DATA_PATH}/ptbxl_database.csv')
    scp = pd.read_csv(f'{DATA_PATH}/scp_statements.csv')
    scp_code_col = scp.columns[0]

    # Define label mapping (consistent with load_and_split_data)
    mi_codes = scp[(scp['diagnostic_subclass'] == 'MI') |
                   (scp['description'].str.contains('infarct', case=False))][scp_code_col].tolist()
    ischemia_codes = scp[scp['diagnostic_subclass'].isin(['STTC', 'NST_', 'ISC_'])][scp_code_col].tolist()
    normal_code = 'NORM'

    signals, labels, patient_ids = [], [], []

    for _, row in metadata.iterrows():
        scp_codes = eval(row['scp_codes'])

        # Check if condition matches
        if condition not in scp_codes:
            continue

        # Get label (same logic as load_and_split_data)
        label = 2 if any(code in mi_codes for code in scp_codes) else \
            1 if any(code in ischemia_codes for code in scp_codes) else \
                0 if normal_code in scp_codes else None

        if label is None:
            continue

        # Load and resample ECG
        record = wfdb.rdrecord(f"{DATA_PATH}/{row['filename_hr']}")
        resampled = np.array([resample(lead, TARGET_SAMPLES) for lead in record.p_signal.T])

        signals.append(resampled)
        labels.append(label)
        patient_ids.append(row['patient_id'])

        if len(signals) >= max_samples:
            break

    # Convert to numpy arrays
    signals = np.array(signals)
    labels = np.array(labels)

    # Return as ECGDataset (same format as load_and_split_data)
    return ECGDataset(signals, labels, augment=augment)