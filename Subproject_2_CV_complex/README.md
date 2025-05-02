# Project README file

This file summarizes all the steps to follow to replicate and use the project.
For each script, you can run the help to know which options are available.

`python -m scripts.train_model --help`

You have first to install the dependencies:
`pip install -r requirements.txt`


**0. Create the dataset**

The CIC-IDS dataset is very big (few hundreds of GB). We have extracted the relevant information from it, but we leave to the user the possibility to implement his own data extraction method.
In order to recreate the dataset we have used, you can simply run the following script:

`python -m scripts.create_final_dataset`

By default, 2 new files will be created in the /data/final folder:
    - test.parquet: this is the dataset which will be used to evaluate the model
    - train.parquet: This is the training dataset, which went through SMOTE algorithm to correct class imbalances.

If the user wants to use another preprocessed dataset, please follow the steps below:
- create a subfolder in the data folder
- copy your train and test datasets there (as parquet files)
- update the yaml file dataset_config.yml to declare the dataset

**1. Add a new model**

The project comes with a pre defined model called BinaryIDSModel.
If you were to add a new model, just follow the following steps:
    - create a new model file in the models folder
    - Make your model derive from the BaseModel class
    - Implement at least the _init_ and forward function
    - Add the @register_model decorator

We have added a model factor to the project, so your project should be automatically detected.


**2. Train a model**

To train a model, use the following script:

`python -m scripts.train_model`

You can add your own arguments to chose the model, the batch size, and the epoch for example.
The available argurments are:
```
--model_name: Model to train
--epochs: Number of epochs
--batch_size: Batch size
--plots: Do you want to plot the epoch losses
--output_name: Name of the file to be saved
--dataset: Name of the dataset (by default: default)
--save_all: Do you want to save models for each epoch (by default: True)
```

**3. Evaluate a model**

To evaluate a model after training, use the following script:

`python -m scripts.evaluate_model`

You can add your own arguments to chose the model, the batch size, and the epoch for example.
The available argurments are:
```
--model_name: Model to load
--batch_size: Batch size
--dataset: Name of the dataset (by default: default)
--model_file: Name of the model file to load (by default: model_state_dict)
--random_model: Set it to True to see the result of a non trained model (by default: False)
--attention_map: Set it to true to calculate attention map (Be ready to wait for 2H) (by default: False)
```

