import torch
from torch.utils import data
from .preprocessing import preprocess
import os

class SliceDataset(data.Dataset):
    """
        Used only for CNN training
    """
    def __init__(self, label_df, target_slices, DATA_DIR, do_preprocess=True):
        self.do_preprocess = do_preprocess
        self.DATA_DIR = DATA_DIR
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
        X = torch.load(os.path.join(self.DATA_DIR, f"{patient_nr}_{slice_nr}.pt"))
        
        return preprocess(X) if self.do_preprocess else X, y