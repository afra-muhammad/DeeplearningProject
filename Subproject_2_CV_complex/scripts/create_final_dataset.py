import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import argparse
from .plot_utils import graph_attack_packets_headers, graph_attack_packets_entropy, plot_data_imabalance


def load_rawdata():
    """Data to load the raw data from parquet files

    Returns
    -------
    pd.Dataframe
        Dataframe holding the data
    """
    # Extract the files from this directory
    dirpath = './data/'
    files = [os.path.join(dirpath, f) for f in tqdm(os.listdir(dirpath), desc="Loading raw parquet files") 
        if os.path.isfile(os.path.join(dirpath, f))]


    df = []

    for f in tqdm(files, desc="Processing raw data files"):
        df.append(pd.read_parquet(f))

    data = pd.concat(df, axis=0)

    return data





def create_final_dataset(data):
    """Creating the final dataset by removing columns which do not change

    """

    header_columns = [col for col in data.columns if col.startswith('Header_bit_')]
    headers_data = data.loc[:, header_columns]

    std_values = headers_data.std()

    # Select columns where std is not 0 (or very close to 0 for floating point precision)
    non_constant_columns = std_values[std_values > 1e-10].index  # Adjust threshold as needed

    print(f'The number of Headers bits have been sized down from 144 to {len(non_constant_columns)}')

    selected_columns = list(non_constant_columns) + ['Entropy_1', 'Entropy_2', 'Entropy_3', 'Entropy_4', 'Entropy_5', 'Entropy_6', 'Entropy_7', 'Entropy_8', 'Entropy_9', 'Entropy_10',  'PayloadEncoding', 'PayloadStrings',
       'Attack', 'Label']
    data_eda = data.loc[:, selected_columns]

    return data_eda




def balance_dataset(data):
    """Using SMOTE to balance the dataset
    """
    # We first encode the label
    data_copy = data.drop(['Attack','PayloadEncoding','PayloadStrings'], axis=1).copy(deep=True)
    data_copy.loc[:,'Label'] = data_copy.loc[:,'Label'].map({'Benign': 0, 'Malicious': 1}).astype('uint8')

    # Separate features and target
    X = data_copy.drop('Label', axis=1)
    y = data_copy['Label']

    # Split data into train and test sets first (important to avoid data leakage)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # The y_train comes back as object which makes SMOTE fail
    y_train = y_train.astype('uint8')
    y_test =  y_test.astype('uint8')

    # Apply SMOTE only to the training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    smote_data_train = pd.concat([X_train_smote, y_train_smote], axis=1)
    data_test = pd.concat([X_test, y_test], axis=1)

    return smote_data_train, data_test

def complete_data(data):
    """Create the final dataset by collating the headers bits with the entropy data

    Parameters
    ----------
    data : input dataset

    Returns
    -------
    Final dataset
    """
    # Make a copy
    data_copy= data.copy(deep=True)
    
    # Set 2 new columns to 0
    data_copy.loc[:,'Dummy1']=0
    data_copy.loc[:,'Dummy2']=0
    
    # Extract the entropy column
    entropy_columns = [c for c in data_copy.columns if c.startswith('Entropy')]

    # Extract the headers columns
    header_columns = [c for c in data_copy.columns if c.startswith('Header')]
    dummy1_columns = [c for c in data_copy.columns if c.startswith('Dummy1')]
    dummy2_columns = [c for c in data_copy.columns if c.startswith('Dummy2')]

    # Create a new dataframe with the columns in the correct order
    data_copy = pd.concat([ data_copy.loc[:, header_columns],
                  data_copy.loc[:, dummy1_columns], 
                  data_copy.loc[:, entropy_columns], 
                  data_copy.loc[:, dummy2_columns],
                data_copy.loc[:, 'Label']], axis=1)
    return data_copy


def create_clean_dataset(args):
    """General function to create the final dataset
    """
    #Loading the raw dataset
    df = load_rawdata()

    # Getting the names from the arguments
    train_name = args.train_name
    test_name = args.test_name

    # Plot the attack packets and entropy
    graph_attack_packets_headers(df)
    graph_attack_packets_entropy(df)

    # Create the final dataset
    final_dataset = create_final_dataset(df)

    # Plot the dataset imbalances
    plot_data_imabalance(final_dataset)

    # Balance the dataset
    train_data, test_data = balance_dataset(final_dataset)

    # Plot the imbalance in the train data after SMOTE
    plot_data_imabalance(train_data)

    # Complete the data
    smote_data_train = complete_data(train_data)
    data_test = complete_data(test_data)

    # Save the clean dataset
    smote_data_train.to_parquet(f'./data/final/{train_name}.parquet', compression='snappy')
    data_test.to_parquet(f'./data/final/{test_name}.parquet', compression='snappy')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model training with arguments')
    parser.add_argument('--train_name', type=str, default='train', help='Name of the final train dataset')
    parser.add_argument('--test_name', type=str, default='test', help='Name of the final test dataset')

    # Parse the arguments passed after -m scripts.test
    args = parser.parse_args()

    create_clean_dataset(args)
