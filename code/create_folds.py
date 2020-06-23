### Imports ###
import os
import pickle
import pandas as pd

### Local imports ###
from dataset.patient_dataframes import get_patient_train_val_dataframes
from dataset.cross_validation import get_train_and_val

### Directories ###
DATA_DIR = '../data/train'
CV_DIR = '../cv'
if not os.path.exists(CV_DIR):
    os.makedirs(CV_DIR)

### Read data ###
label_df = pd.read_csv(os.path.join(DATA_DIR, "labels_slices.csv"), names=["patient_nr", "slice_nr", "class"])

### Get folds ###
_, _, _, _, folds = get_patient_train_val_dataframes(label_df, k=5)

### Get fold data ###
train_dfs = []
val_dfs = []
train_patients = []
val_patients = []

for val_fold in range(len(folds)):
    train_patient, val_patient = get_train_and_val(folds, val_fold)
    train_df = label_df[label_df["patient_nr"].isin(train_patient)]
    val_df = label_df[label_df["patient_nr"].isin(val_patient)]

    train_patients.append(train_patient)
    val_patients.append(val_patient)
    train_dfs.append(train_df)
    val_dfs.append(val_df)

### Save fold data ###
with open(CV_DIR + "/train_dfs.pkl", "wb") as f:
    pickle.dump(train_dfs, f)
with open(CV_DIR + "/val_dfs.pkl", "wb") as f:
    pickle.dump(val_dfs, f)
with open(CV_DIR + "/train_patients.pkl", "wb") as f:
    pickle.dump(train_patients, f)
with open(CV_DIR + "/val_patients.pkl", "wb") as f:
    pickle.dump(val_patients, f)