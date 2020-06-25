import os
os.system("python -m wandb.cli login 8d7601a3f5545dac156785dbc02523182dcf0458")

### Import packages ###
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import models
import pandas as pd
import numpy as np
import wandb
import argparse
import sys

### Local imports ###
# Necessary for local imports
sys.path.append("..")
from dataset.patient_dataset import PatientDataset
from train import Trainer
from models.lstm import LSTM
from models.omnipotent_resnet import Net
from models.combined_net import CombinedNet

### DEFAULT PARAMETERS ###
### Data parameters ###
DATA_DIR = '../../../sliced_data/train'
CNN_LOC = '../../../models/resnet34_standardised_011.pt'
DS_DIR = '../../../data_split'
TARGET_SLICES = (0, 32)  # The slices we will train on for each patient
### Model parameters ###
MODEL_DIR = '../models'  # Directory where best models are saved
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Train on GPU or CPU
N_FEATURES = 128  # The length of feature vectors that the CNN outputs/LSTM will use
### Train parameters ###
VAL_FILENAME = "val_df_lstm.csv"
EPOCHS = 15
BATCH_SIZE = 16
LR = 0.0001

### Argument parser ###
parser = argparse.ArgumentParser(description='Train a specified ResNet model.')
parser.add_argument('name', type=str, help="Name of the pre-trained model")
parser.add_argument('-d', type=str, nargs='?', dest="data_dir",
                    default=DATA_DIR, help="Path to directory with data")
parser.add_argument('-c', type=str, nargs='?', dest="cnn_loc",
                    default=CNN_LOC, help="Path to the ResNet weights file")
parser.add_argument('-lr', type=float, nargs='?', dest="learning_rate",
                    default=LR, help='Learning rate')
parser.add_argument('-e', type=int, nargs='?', dest="epochs",
                    default=EPOCHS, help='Number of epochs')
parser.add_argument('-b', type=int, nargs='?', dest="batch_size",
                    default=BATCH_SIZE, help="Batch size")
parser.add_argument('-m', type=str, nargs='?', dest="model_dir",
                    default=MODEL_DIR, help="Where models will be saved")
parser.add_argument('-f', type=int, nargs='?', dest="n_features",
                    default=N_FEATURES, help="Number of output features of last FC layer")
parser.add_argument('-s', nargs='+', dest='target_slices',
                    default=TARGET_SLICES, help="Which slices to use for training")
parser.add_argument('-v', type=str, nargs='?', dest="val_filename",
                    default=VAL_FILENAME, help="Where to store the validation dataframe (for failure analysis)")
parser.add_argument('--tuple', action="store_true", dest="is_target_tuple",
                    help="Whether slices argument is tuple or not")
parser.add_argument('--pretrained', action="store_true", help="Whether networks are pretrained")

if __name__ == "__main__":
    args = parser.parse_args()

    # Load and check data
    train_df = pd.read_csv(os.path.join(DS_DIR, "train_df.csv"), names=["patient_nr", "slice_nr", "class"])
    print(f"\nNumber of unique patient numbers in training set: {len(np.unique(train_df['patient_nr']))}")
    print(f"Number of unique slice numbers in training set:   {len(np.unique(train_df['slice_nr']))}")
    print(f"Number of unique class values in training set:    {len(np.unique(train_df['class']))}")
    val_df = pd.read_csv(os.path.join(DS_DIR, "val_df.csv"), names=["patient_nr", "slice_nr", "class"])
    print(f"\nNumber of unique patient numbers in validation set: {len(np.unique(val_df['patient_nr']))}")
    print(f"Number of unique slice numbers in validation set:   {len(np.unique(val_df['slice_nr']))}")
    print(f"Number of unique class values in validation set:    {len(np.unique(val_df['class']))}")
    
    train_patients = pd.read_csv(os.path.join(DS_DIR, "train_patients.csv"), names=["patient_nr"]).to_numpy().flatten()
    print(f"Number of patient numbers in the train patients list:      {len(train_patients['patient_nr'])}")
    val_patients = pd.read_csv(os.path.join(DS_DIR, "val_patients.csv"), names=["patient_nr"]).to_numpy().flatten()
    print(f"Number of patient numbers in the validation patients list: {len(val_patients['patient_nr'])}")

    # Load in model
    model = models.resnet34(pretrained=args.pretrained)
    resnet = Net(model, args.name, args.n_features)
    resnet.load_state_dict(torch.load(args.cnn_loc))
    lstm_net = LSTM(n_features=args.n_features, n_hidden=64, n_layers=2)
    combined_net = CombinedNet(name=args.name, cnn_net=resnet, lstm_net=lstm_net)
    combined_net.set_learning_cnn_net(False)

    ### Loss and optimizer ###
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(combined_net.parameters(), lr=args.learning_rate)

    ### Create dataframes for training and validation ###
    # Set correct target slices
    if args.is_target_tuple:
        args.target_slices = tuple(args.target_slices)

    # Set train/validation loaders
    train_set = PatientDataset(train_df, train_patients, args.target_slices, args.data_dir, DEVICE)
    val_set = PatientDataset(val_df, val_patients, args.target_slices, args.data_dir, DEVICE)

    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # Initialise W&B settings
    train_percentage = float(len(train_patients['patient_nr'])) / float(len(train_patients['patient_nr']) + len(val_patients['patient_nr']))
    wandb.init(project="braintriage")
    wandb.config.update({"model_type": args.name, "epochs": args.epochs, "batch_size": args.batch_size,
                         "learning_rate": args.learning_rate,
                         "n_features": args.n_features, "target_slices": args.target_slices,
                         "is_target_tuple": args.is_target_tuple,
                         "train_percentage": train_percentage})
    wandb.watch(combined_net)
    trainer = Trainer(model=combined_net, criterion=criterion, optimizer=optimizer, device=DEVICE,
                      train_loader=train_loader, val_loader=val_loader, n_epochs=args.epochs, model_dir=args.model_dir)
    trainer.train_and_validate()
