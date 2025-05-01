import pandas as pd

#loading the first set of 10 files
parquet_files = [f"data/output_part_{i}.parquet" for i in range(11)]
dfs = [pd.read_parquet(file) for file in parquet_files]
df = pd.concat(dfs, ignore_index=True)

#loading the second set of 10 files
parquet_files1 = [f"data/output_part_{i}.parquet" for i in range(11,21)]
dfs1 = [pd.read_parquet(file) for file in parquet_files1]
df1 = pd.concat(dfs1, ignore_index=True)

#creating a full dataset using the Pandas concat function.
dfs=[df,df1]
dff=pd.concat(dfs)
print(dff)

#exporting the dataframe as a csv file
dff.to_csv('output_data/data_for_read.csv', index=False)