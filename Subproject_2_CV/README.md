#  Binary Intrusion Detection System (IDS) using TCP/IP Header Bits

This project implements a lightweight convolutional neural network (CNN) based IDS for detecting malicious packets using binary image representations of TCP/IP headers by following the resarch paper  *Intrusion Detection using TCP/IP Single Packet Header Binary Image*, applying it to a preprocessed subset of the CSE-CIC-IDS2018 dataset.

---

##  Project Overview

- **Goal**: Classify each packet as `Benign` or `Attack` using only TCP/IP header bits.
- **Technique**: Convert the first 144 header bits into 12×12 binary images for CNN input.
- **Data Source**: 
  - CSE-CIC-IDS2018 Dataset
  - Cleaned and balanced using `clean_dataset.py`

---

##  Project Pipeline

### 1. **Prepare Environment**
- Clone the repo and install dependencies
- Create a Conda environment:
    ```bash
    conda env create -f environment.yml
    conda activate ids-cnn-env
    ```

### 2. **Download and Clean Dataset**
- Place all `.parquet` files into a folder (e.g. `./Parquet files`) present in Data folder on github.
- Run the data cleaner:
    ```bash
    python clean_dataset.py
    ```
- Output: `cleaned_balanced_dataset.parquet`

### 3. **Train Binary IDS CNN**
- Open and run the notebook:
    ```bash
    IDS_Binary classification.ipynb
    ```
- Performs preprocessing + training using 5-Fold Stratified Cross-Validation


---

##  Outputs

- `cleaned_balanced_dataset.parquet` — balanced and binary-labeled dataset
- `IDS_Binary classification.ipynb` — full pipeline with training and evaluation
- Visuals:
  - `confusion_matrix.png`
  - `loss_curve.png`
  - `header_image_samples.png`

---

##  Requirements

Install using Conda:
```bash
conda env create -f environment.yml
conda activate ids-cnn-env
```

Or with pip:
```bash
pip install -r requirements.txt
```

---

---

##  Credits

Developed using TensorFlow and the CSE-CIC-IDS2018 dataset.
