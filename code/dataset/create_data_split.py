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
train_df, val_df, train_patients, val_patients, folds = get_patient_train_val_dataframes(label_df, k=FOLDS)

### Save fold data ###
with open(DS_DIR + "/train_df.pkl", "wb") as f:
    pickle.dump(train_df, f)
with open(DS_DIR + "/val_df.pkl", "wb") as f:
    pickle.dump(val_df, f)
with open(DS_DIR + "/train_patients.pkl", "wb") as f:
    pickle.dump(train_patients, f)
with open(DS_DIR + "/val_patients.pkl", "wb") as f:
    pickle.dump(val_patients, f)