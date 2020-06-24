
import numpy as np

def get_slice_train_val_dataframes(label_df, train_percentage=0.8):
    """
        Generates an evenly class-distributed train/val split for all slices in the dataset, 
        regardless of the patient number.
    """
    # Sort values on class, because we know the first half is 0 and the second half is 1
    temp_label_df = label_df.sort_values("class")

    n_slices_per_class = temp_label_df.shape[0]//2
    normal_indices = np.random.permutation(n_slices_per_class)
    abnormal_indices = np.random.permutation(n_slices_per_class) + n_slices_per_class

    # Split the patient lists into train and validation (index must be in list to split in two parts)
    split_idx = int(train_percentage * n_slices_per_class)
    normal_train_indices, normal_val_indices = np.split(normal_indices, [split_idx])
    abnormal_train_indices, abnormal_val_indices = np.split(abnormal_indices, [split_idx])

    # Append train and validation dataframes 
    train_df = temp_label_df.iloc[normal_train_indices].append(temp_label_df.iloc[abnormal_train_indices])
    val_df = temp_label_df.iloc[normal_val_indices].append(temp_label_df.iloc[abnormal_val_indices])

    return train_df, val_df
   
label_df = pd.read_csv(os.path.join(WAAR_STAAT_DATA, "labels_slices.csv"), names=["patient_nr", "slice_nr", "class"])
label_df["class"] = label_df["class"].astype("int8")

train_df, val_df = get_slice_train_val_dataframes(label_df, train_percentage = 0.9)
train_patients = np.unique(train_df["patient_nr"])
val_patients = np.unique(val_df["patient_nr"])
intersect = np.intersect(train_patients, val_patients)
print(intersect)
