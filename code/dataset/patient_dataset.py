import torch
from torch.utils import data
from .preprocessing import preprocess

import os

class PatientDataset(data.Dataset):
    """
        Used for combination training, so 
            (1) LSTM separately and 
            (2) finetuning phase of CNN and LSTM together
    """
    def __init__(self, label_df, patient_list, target_slices,DATA_DIR,DEVICE):
        self.DATA_DIR = DATA_DIR
        self.DEVICE = DEVICE
        self.label_df = label_df
        self.patient_list = patient_list
        self.target_slices = target_slices
        
    def __len__(self):
        return self.patient_list.shape[0]
    
    def __getitem__(self, index):
        patient_nr = self.patient_list[index]
        
        if type(self.target_slices) == tuple:
            slice_list = [self.get_slice_tensor(patient_nr, slice_nr) for slice_nr in range(*self.target_slices)]
        elif type(self.target_slices) == list:
            slice_list = [self.get_slice_tensor(patient_nr, slice_nr) for slice_nr in self.target_slices]
        else: # If single integer
            slice_list = [self.get_slice_tensor(patient_nr, self.target_slices)]
        
        X = torch.stack(slice_list)
        y = self.label_df[self.label_df["patient_nr"] == patient_nr].iloc[0,2]
        
        return X, y
    
    def get_slice_tensor(self, patient_nr, slice_nr):
        return preprocess(torch.load(os.path.join(self.DATA_DIR, f"{patient_nr}_{slice_nr}.pt"), map_location=self.DEVICE))