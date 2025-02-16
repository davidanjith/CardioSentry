import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Load raw and interpolated data
raw_parquet_file = "processed_data/ppg_dalia_cleaned/raw_combined_ppg_dalia.parquet"
interpolated_parquet_file = "processed_data/ppg_dalia_cleaned/combined_ppg_dalia.parquet"

raw_df = pd.read_parquet(raw_parquet_file)
interpolated_df = pd.read_parquet(interpolated_parquet_file)

# Ensure both datasets have the same indices
if not raw_df.index.equals(interpolated_df.index):
    print("Warning: Indices are not aligned! Aligning data...")
    interpolated_df = interpolated_df.reindex(raw_df.index, method='nearest')

# Sample Data
sample_size = 1000
sample_indices = np.random.choice(len(raw_df), sample_size, replace=False)
sample_indices.sort()  # Ensure order is preserved

time = np.linspace(0, 1, sample_size)
original_ppg = raw_df["PPG"].iloc[sample_indices]
interpolated_ppg = interpolated_df["PPG"].iloc[sample_indices]

# Apply improved interpolation
ppg_interpolator = interp1d(raw_df.index, raw_df["PPG"], kind='cubic', fill_value="extrapolate")
interpolated_df["PPG"] = ppg_interpolator(interpolated_df.index)

# Ensure activity labels remain discrete
if "Activity" in raw_df.columns and "Activity" in interpolated_df.columns:
    interpolated_df["Activity"] = raw_df["Activity"].ffill().bfill()  # Forward-fill and back-fill

# Re-sample after interpolation
interpolated_ppg = interpolated_df["PPG"].iloc[sample_indices]
interpolated_activity = interpolated_df["Activity"].iloc[sample_indices]
original_activity = raw_df["Activity"].iloc[sample_indices]

# Plot PPG Before vs After Interpolation
plt.figure(figsize=(12, 4))
plt.plot(time, original_ppg, label='Original PPG', alpha=0.7)
plt.plot(time, interpolated_ppg, label='Interpolated PPG', linestyle='dashed', alpha=0.7)
plt.xlabel("Time (normalized)")
plt.ylabel("PPG Signal")
plt.title("PPG Before vs After Improved Interpolation")
plt.legend()
plt.show()

# Statistical Check for PPG
print("Original PPG Mean:", original_ppg.mean(), "Std:", original_ppg.std())
print("Interpolated PPG Mean:", interpolated_ppg.mean(), "Std:", interpolated_ppg.std())

# Plot Activity Labels Before vs After Interpolation
plt.figure(figsize=(12, 4))
plt.plot(time, original_activity, label='Original Activity', alpha=0.7)
plt.plot(time, interpolated_activity, label='Interpolated Activity', linestyle='dashed', alpha=0.7)
plt.xlabel("Time (normalized)")
plt.ylabel("Activity Level")
plt.title("Activity Labels Before vs After Improved Interpolation")
plt.legend()
plt.show()

# Statistical Check for Activity Labels
print("Original Activity Mean:", original_activity.mean(), "Std:", original_activity.std())
print("Interpolated Activity Mean:", interpolated_activity.mean(), "Std:", interpolated_activity.std())
