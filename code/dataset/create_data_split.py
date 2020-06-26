### Imports ###
import os
import pickle
import pandas as pd
import numpy as np
import argparse

### DEFAULT PARAMETERS ###
### Fold parameter ###
K = 10 # How many folds to split the data into (train percentage = (k-1)/k)
### Directories ###
DATA_DIR = '../../../data_sliced/train'
DS_DIR = '../../../data_split'

parser = argparse.ArgumentParser(description='Split the train data into train and validation sets.')
parser.add_argument("-k", type=int, nargs = "?", dest="k", 
                    default=K, help = "How many folds to split the data into (train percentage = (k-1)/k)")
parser.add_argument("-d", type=str, nargs='?', dest="data_dir",
                    default=DATA_DIR, help="Path to directory with slice data")
parser.add_argument("-ds", type=str, nargs='?', dest="ds_dir",
                    default=DS_DIR, help="Path to directory where split dataframes will be stored")    


def get_patient_train_val_dataframes(label_df, k = 5, val_fold = 0):
    """
        Creates an evenly class-distributed train/val split for all patients in the dataset.
        Target slices are taken into account in the Dataset classes.
        
        With K, you can determine how many folds need to be created.
        With val_fold, you can determine which fold is the validation fold.
            k = 5 and val_fold = 4 means that you have 80% train data and the last 20% is val data 
            (which has been the standard way of splitting the data so far in the project)
    """
    # Generate dataframe that maps patient_nr to class
    temp_label_df = label_df.drop("slice_nr",axis=1).drop_duplicates()
    # Sort values to get patient_list in same order as patient_nr col in dataframe
    temp_label_df = temp_label_df.sort_values("patient_nr")
    patient_list = np.unique(temp_label_df["patient_nr"])
    print(len(patient_list))

    abnormal_patients = patient_list[np.where(temp_label_df["class"].values)[0]]
    # Shuffle patient_list for random train/val split (only in-place)
    np.random.shuffle(abnormal_patients)
    normal_patients = np.setdiff1d(patient_list, abnormal_patients)
    # np.setdiff1d returns a sorted array, so shuffle normal patients too
    np.random.shuffle(normal_patients)
    
    # Create Folds
    folds = create_k_strat_folds(normal_patients, abnormal_patients, k)
    
    # Get Train and Validation IDs
    train_patients, val_patients = get_train_and_val(folds, val_fold)

    # Append train and validation dataframes 
    train_df = label_df[label_df["patient_nr"].isin(train_patients)]
    val_df = label_df[label_df["patient_nr"].isin(val_patients)]
    
    return train_df, val_df, pd.DataFrame(train_patients), pd.DataFrame(val_patients)

def create_k_strat_folds(normal_patients, abnormal_patients, k = 5):
    """
    Creates k stratified folds by creating 5 folds of each array.
    
        - normal_patients: array with IDs of normal classified patients
        - abnormal_patients: array with IDs of abnormal classified patients
        - k: number of folds to create
        
    If np.split cannot create k folds, it will throw an error
    """
    normal_folds = np.split(normal_patients, k)
    abnormal_folds = np.split(abnormal_patients, k)
    folds = np.concatenate((normal_folds, abnormal_folds), axis = 1)
        
    return folds

def get_train_and_val(folds, val_fold = 0):
    """
    From the folds created in create_k_strat_folds, 
    this function returns a train_set with indices that are in training folds
    and a val_set with indices from the validation fold.
    
        - folds: 2D-array with folds created in create_k_strat_folds
        - val_fold: integer that indicates which fold is the validation fold
        
    Keep in mind that val_fold cannot be larger than k in create_k_strat_folds!!
    """
    val_set = folds[val_fold]
    train_set= folds[np.arange(len(folds))!=val_fold]
    
    return train_set.flatten(), val_set


if __name__ == "__main__":              
    args = parser.parse_args()

    ### Read data ###
    label_df = pd.read_csv(os.path.join(args.data_dir, "labels_slices.csv"), names=["patient_nr", "slice_nr", "class"])

    ### Get folds ###
    train_df, val_df, train_patients, val_patients = get_patient_train_val_dataframes(label_df, k=args.k)

    ### Save fold data ###
    if not os.path.exists(args.ds_dir):
        os.makedirs(args.ds_dir)
    train_df.to_csv(os.path.join(args.ds_dir, "train_df.csv"), header=False)
    val_df.to_csv(os.path.join(args.ds_dir, "val_df.csv"), header=False)
    train_patients.to_csv(os.path.join(args.ds_dir, "train_patients.csv"), header=False)
    val_patients.to_csv(os.path.join(args.ds_dir, "val_patients.csv"), header=False)
