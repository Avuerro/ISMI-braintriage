import numpy as np
from dataset.cross_validation import create_k_strat_folds
from dataset.cross_validation import get_train_and_val

def get_patient_train_val_dataframes(label_df, train_percentage=0.8, k = 5, val_fold = 4):
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
    
    return train_df, val_df, train_patients, val_patients