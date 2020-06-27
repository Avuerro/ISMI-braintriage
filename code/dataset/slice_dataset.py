import torch
from torch.utils import data
from .preprocessing import preprocess, augment
import os

class SliceDataset(data.Dataset):
    """
        Used only for CNN training
    """
    def __init__(self, label_df, target_slices, data_dir, flip_prob=0, rotate_prob=0):
        self.data_dir = data_dir
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        if type(target_slices) == tuple:
            self.label_df = label_df[label_df["slice_nr"].isin(range(*target_slices))]
        elif type(target_slices) == list:
            self.label_df = label_df[label_df["slice_nr"].isin(target_slices)]
        else:
            self.label_df = label_df[label_df["slice_nr"] == target_slices]
        
    def __len__(self):
        return self.label_df.shape[0]
    
    def __getitem__(self, index):
        patient_nr, slice_nr, cls = self.label_df.iloc[index].to_numpy()
        y = cls
        X = torch.load(os.path.join(self.data_dir, f"{patient_nr}_{slice_nr}.pt"))
        
        return augment(preprocess(X), self.flip_prob, self.rotate_prob), y