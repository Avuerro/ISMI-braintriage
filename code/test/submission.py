import sys
# Necessary for relative import of patient dataset
sys.path.append("..")
import os
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils import data
from torchvision import models

from dataset.patient_dataset import PatientDataset
from models.combined_net import CombinedNet
from models.lstm import LSTM
from models.omnipotent_resnet import Net

### DEFAULT PARAMETERS ###
### Data parameters ###
TEST_DATA_DIR = "../data/test" # The slices we will train on for each patient
TARGET_SLICES = (0,32)
### Model parameters ###
MODEL_DIR = '../models'  # Directory where best models are saved
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Train on GPU or CPU
RESNET_MODEL_TYPE = "resnet34" # Which type of resnet is used by the model
PRETRAINED = False
### Test parameters ###
N_FEATURES = 128
BATCH_SIZE = 2
SUBMISSION_DIR = "../submissions"

### Argument parser ###
parser = argparse.ArgumentParser(description='Create a submission using a specified pretrained model.')
parser.add_argument('name', type=str, help="Name of the model (resnet, lstm, combinednet")
parser.add_argument('resnet', type=str, help = "Type of ResNet to use (resnet18 or resnet34)")
parser.add_argument('filename', type=str, help="Name of the file where the trained model's parameters are stored")
parser.add_argument('-d', type=str, nargs='?', dest="test_data_dir",
                    default=TEST_DATA_DIR, help="Path to directory with test data")
parser.add_argument('-m', type=str, nargs='?', dest="model_dir",
                    default=MODEL_DIR, help="Where model parameters are stored")
parser.add_argument('-sd', type=str, nargs='?', dest="submission_dir",
                    default=SUBMISSION_DIR, help="Where submission will be stored")
parser.add_argument('-b', type=int, nargs='?', dest="batch_size",
                    default=BATCH_SIZE, help="Batch size")
parser.add_argument('-s', nargs='+', dest='target_slices',
                    default=TARGET_SLICES, help="Which slices to use for training")
parser.add_argument('-f', type=int, nargs='?', dest="n_features",
                    default = N_FEATURES, help="Number of output features of last FC layer")                   
    
if __name__ == "__main__":
    args = parser.parse_args()
    
    # Load label dataframe
    label_df = pd.read_csv(os.path.join(args.test_data_dir, "labels_slices.csv"), names=["patient_nr", "slice_nr", "class"])
    
    # Initialize ResNet
    if args.resnet == "resnet34":
        model = models.resnet34()
    elif args.resnet == "resnet18":
        model = models.resnet18()
    else:
        print(f'No resnet with name {args.resnet}')
        exit()

    # Change the Pre-Trained Model to our own Defined Model
    resnet = Net(model, args.name, args.n_features)
    lstm_net = LSTM(n_features=args.n_features, n_hidden=64, n_layers=2)
    combined_net = CombinedNet(name=args.name, cnn_net=resnet, lstm_net=lstm_net)
    combined_net.load_state_dict(torch.load(os.path.join(args.model_dir, args.filename), map_location=DEVICE))
    combined_net.to(DEVICE)

    # Create dataset and dataloader
    patient_list = np.unique(label_df["patient_nr"])
    test_set = PatientDataset(label_df, patient_list, args.target_slices, args.test_data_dir, DEVICE)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count())

    # Run the model on all testing data
    combined_net.eval()
    all_probabilities, all_classes = [], []
    # Batch-wise process all test data (all patient numbers are processed in ascending order)
    for images, _ in tqdm(test_loader):
        images = images.to(DEVICE)
        output = combined_net(images).detach().cpu()
        # Compute probabilities (requirement: round to 5 decimals)
        probabilities = np.round(torch.sigmoid(output).numpy(), 5)
        # Compute class from probability (>0.5 = abnormal)
        classes = (probabilities > 0.5).astype(np.uint8)
        all_probabilities.extend(probabilities)
        all_classes.extend(classes)
    
    # Create submission dataframe and write it to an output file
    submission = pd.DataFrame({"case":patient_list, "probability":all_probabilities, "class":all_classes})
    
    if not os.path.exists(args.submission_dir):
        os.makedirs(args.submission_dir)
    submission.to_csv(os.path.join(args.submission_dir, args.filename + "_submission.csv"), index=False)
    