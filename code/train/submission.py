import sys
# Necessary for relative import of patient dataset
sys.path.append("..")
import os
import torch
import numpy as np
import pandas as pd
from torch.utils import data
from tqdm.notebook import tqdm
from dataset.patient_dataset import PatientDataset

TEST_DATA_PATH = "../data/test"

def create_submission(net, batch_size, device):
    # Load label dataframe
    label_df = pd.read_csv(os.path.join(TEST_DATA_PATH, "labels_slices.csv"), names=["patient_nr", "slice_nr", "class"])
    # Create dataset and dataloader
    patient_list = np.unique(label_df["patient_nr"])
    test_set = PatientDataset(label_df, patient_list, (0,31), TEST_DATA_PATH, device)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())

    net.eval()
    all_probabilities, all_classes = [], []
    # Batch-wise process all test data (all patient numbers are processed in ascending order)
    for images, _ in tqdm(test_loader):
        images = images.to(device)
        output = net(images).detach().cpu()
        # Compute probabilities (requirement: round to 5 decimals)
        probabilities = np.round(torch.sigmoid(output).numpy(), 5)
        # Compute class from probability (>0.5 = abnormal)
        classes = (probabilities > 0.5).astype(np.uint8)
        all_probabilities.extend(probabilities)
        all_classes.extend(classes)
    
    # Create submission dataframe
    submission = pd.DataFrame({"case":patient_list, "probability":all_probabilities, "class":all_classes})
    
    return submission