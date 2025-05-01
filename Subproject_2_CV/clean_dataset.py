import pandas as pd
import glob
import os

# ========== CONFIGURATION SECTION ==========
# Folder containing all .parquet files
parquet_folder = "C:\\project1\\Parquet files"  

# Name of the output file to save the cleaned and balanced dataset
output_file = "cleaned_balanced_dataset.parquet"

# Maximum number of samples to keep per class (for balancing)
max_per_class = 15000

# Define which attack types we want to keep in the final dataset
target_attacks = {
    "SQL Injection",
    "BruteForce-Web",
    "BruteForce-XSS",
    "DoS-Slowloris",
    "DoS-GoldenEye",
    "Infiltration",
    "Benign"  # benign traffic
}

# Header bit columns to extract (these form the 12x12 binary image)
header_cols = [f"Header_bit_{i}" for i in range(1, 145)]
# ===========================================

all_rows = []

# Find all .parquet files in the given folder
parquet_files = glob.glob(os.path.join(parquet_folder, "*.parquet"))
print(f"Found {len(parquet_files)} parquet files.")

for i, file in enumerate(parquet_files):
    print(f" [{i+1}/{len(parquet_files)}] Loading: {file}")
    try:
        df = pd.read_parquet(file)
    except Exception as e:
        print(f" Failed to read {file}: {e}")
        continue

    # Ensure that the 'Attack' column is present
    if 'Attack' not in df.columns:
        print(" No 'Attack' column — skipping.")
        continue

    # Clean the 'Attack' column and keep only target attacks
    df['Attack'] = df['Attack'].fillna("Unknown").str.strip()
    df = df[df['Attack'].isin(target_attacks)]

    if df.empty:
        print(" No matching attacks in this file — skipping.")
        continue

    # Binary label: 0 for Benign, 1 for any attack
    df['Label'] = df['Attack'].apply(lambda x: 0 if x == "Benign" else 1)

    # Check if all required header bit columns are present
    missing_bits = [col for col in header_cols if col not in df.columns]
    if missing_bits:
        print(f" Missing header bit columns in {file}: {missing_bits[:5]}... — skipping.")
        continue

    # Keep only relevant columns: header bits + attack type + binary label
    df = df[header_cols + ['Attack', 'Label']]

    # Optimize memory by downcasting bit columns to unsigned int (0/1)
    df[header_cols] = df[header_cols].apply(pd.to_numeric, downcast='unsigned')

    all_rows.append(df)

# Stop if no valid data was collected
if not all_rows:
    print(" No valid data collected — exiting.")
    exit()

# Combine all data into a single DataFrame
df_all = pd.concat(all_rows, ignore_index=True)
print(f"\n Combined shape: {df_all.shape}")
print(df_all['Attack'].value_counts())

# Balance the dataset: limit each attack type to max_per_class rows
print(f"\n Balancing classes to max {max_per_class} each...")
df_balanced = (
    df_all.groupby("Attack")
    .apply(lambda x: x.sample(n=min(len(x), max_per_class), random_state=42))
    .reset_index(drop=True)
)

print(f"\n Final balanced shape: {df_balanced.shape}")
print(df_balanced['Attack'].value_counts())

# Save the final cleaned and balanced dataset to a new parquet file
df_balanced.to_parquet(output_file)
print(f"\n Saved cleaned dataset to: {output_file}")
