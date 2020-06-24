### Imports ###
import os
import pickle
import pandas as pd

### Local imports ###
from dataset.patient_dataframes import get_patient_train_val_dataframes
from dataset.cross_validation import get_train_and_val

### Parameters ###
FOLDS = 10

### Directories ###
DATA_DIR = '../../../sliced_data/train'
DS_DIR = '../../../data_split'
if not os.path.exists(DS_DIR):
    os.makedirs(DS_DIR)

### Read data ###
label_df = pd.read_csv(os.path.join(DATA_DIR, "labels_slices.csv"), names=["patient_nr", "slice_nr", "class"])

### Get folds ###
train_df, val_df, train_patients, val_patients = get_patient_train_val_dataframes(label_df, k=FOLDS)

### Save fold data ###
train_df.to_csv(os.path.join(DS_DIR, "train_df.csv"))
val_df.to_csv(os.path.join(DS_DIR, "val_df.csv"))
train_patients.to_csv(os.path.join(DS_DIR, "train_patients.csv"))
val_patients.to_csv(os.path.join(DS_DIR, "val_patients.csv"))