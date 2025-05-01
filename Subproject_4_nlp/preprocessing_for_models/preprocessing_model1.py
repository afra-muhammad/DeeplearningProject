import os
import pandas as pd

pd.set_option('future.no_silent_downcasting', True)
export_folder = "../output_data"
df=pd.read_csv('../output_data/data_for_read.csv')

#replace 'Benign' with 0 and 'Malicious' by 1
df['Label'] = df['Label'].replace({'Benign': 0, 'Malicious': 1})

# Prepare the features/lists
entropy_features = df.iloc[:, :10].values
tcp_features = df.iloc[:, 10:154].values
payloads = df['PayloadStrings'].fillna("<EMPTY>").astype(str)
labels = df['Label'].values

# Convert and export each as CSV
pd.DataFrame(entropy_features).to_csv(f"{export_folder}/entropy_features.csv", index=False, header=False)
pd.DataFrame(tcp_features).to_csv(f"{export_folder}/tcp_features.csv", index=False, header=False)
payloads.to_csv(f"{export_folder}/payloads.csv", index=False, header=True)
pd.DataFrame(labels, columns=["Label"]).to_csv(f"{export_folder}/labels.csv", index=False)

print("Export completed.")