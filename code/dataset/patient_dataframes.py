import numpy as np

def get_patient_train_val_dataframes(label_df, train_percentage=0.8):
    """
        Creates an evenly class-distributed train/val split for all patients in the dataset.
        Target slices are taken into account in the Dataset classes.
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

    # Split the patient lists into train and validation (index must be in list to split in two parts)
    n_patients_per_class = patient_list.shape[0]/2
    split_idx = int(train_percentage * n_patients_per_class)
    normal_train_patients, normal_val_patients = np.split(normal_patients, [split_idx])
    abnormal_train_patients, abnormal_val_patients = np.split(abnormal_patients, [split_idx])

    # Append train and validation dataframes 
    train_df = label_df[label_df["patient_nr"].isin(normal_train_patients)] \
                .append(label_df[label_df["patient_nr"].isin(abnormal_train_patients)])
    val_df = label_df[label_df["patient_nr"].isin(normal_val_patients)] \
                .append(label_df[label_df["patient_nr"].isin(abnormal_val_patients)])

    # Create training and validation patient lists
    train_patients = np.append(normal_train_patients, abnormal_train_patients)
    val_patients = np.append(normal_val_patients, abnormal_val_patients)
    
    return train_df, val_df, train_patients, val_patients