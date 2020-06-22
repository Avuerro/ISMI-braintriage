import os
os.system("python -m wandb.cli login 8d7601a3f5545dac156785dbc02523182dcf0458")
### Import packages ###
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import models
import pandas as pd
import numpy as np
import wandb
import argparse

### Local imports ###
from dataset.slice_dataframes import get_slice_train_val_dataframes
from dataset.patient_dataframes import get_patient_train_val_dataframes
from dataset.slice_dataset import SliceDataset
from dataset.patient_dataset import PatientDataset
from train.train import Trainer
from models.lstm import LSTM
from models.omnipotent_resnet import Net
from models.combined_net import CombinedNet

### DEFAULT PARAMETERS ###
### Data parameters ###
DATA_DIR = '../data/train'
LSTM_DIR = '../models/lstm_000.pt'
TARGET_SLICES = (0, 32)  # The slices we will train on for each patient
TRAIN_PERCENTAGE = 0.9  # Percentage of data that will be used for training
### Model parameters ###
MODEL_DIR = '../models'  # Directory where best models are saved
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Train on GPU or CPU
N_FEATURES = 128  # The length of feature vectors that the CNN outputs/LSTM will use
### Train parameters ###
EPOCHS = 30
BATCH_SIZE = 16
LR = 0.0001

### Argument parser ###
parser = argparse.ArgumentParser(description='Train a specified ResNet model.')
parser.add_argument('name', type=str, help="Name of the pre-trained model")
parser.add_argument('-d', type=str, nargs='?', dest="data_dir",
                    default=DATA_DIR, help="Path to directory with data")
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
parser.add_argument('-tp', type=float, nargs='?', dest="train_percentage",
                    default=TRAIN_PERCENTAGE, help="Percentage of data to use for training")
parser.add_argument('--tuple', action="store_true", dest="is_target_tuple",
                    help="Whether slices argument is tuple or not")
parser.add_argument('--pretrained', action="store_true", help="Whether networks are pretrained")

if __name__ == "__main__":
    args = parser.parse_args()

    # Load and check data
    label_df = pd.read_csv(os.path.join(args.data_dir, "labels_slices.csv"), names=["patient_nr", "slice_nr", "class"])
    label_df["class"] = label_df["class"].astype("int8")
    patient_list = np.unique(label_df["patient_nr"])
    print(label_df.head(), f"Dataframe shape: {label_df.shape}", sep="\n")
    print(f"\nNumber of unique patient numbers: {len(np.unique(label_df['patient_nr']))}")
    print(f"Number of unique slice numbers:   {len(np.unique(label_df['slice_nr']))}")
    print(f"Number of unique class values:    {len(np.unique(label_df['class']))}")

    # Load in model
    model = models.resnet34(pretrained=args.pretrained)
    resnet = Net(model, args.name, args.n_features)
    lstm_net = LSTM(n_features=args.n_features, n_hidden=64, n_layers=2)
    combined_net = CombinedNet(name=args.name, cnn_net=resnet, lstm_net=lstm_net)
    combined_net.load_state_dict(torch.load(LSTM_DIR))
    combined_net.set_learning_cnn_net(True)

    ### Loss and optimizer ###
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(combined_net.parameters(), lr=args.learning_rate)

    ### Create dataframes for training and validation ###
    # Take 5 folds
    _, _, _, _, folds = get_patient_train_val_dataframes(label_df, k=5)

    # Store information for all folds
    # TODO: Store as files
    train_dfs = []
    val_dfs = []
    train_patients = []
    val_patients = []

    # Set correct target slices
    if args.is_target_tuple:
        args.target_slices = tuple(args.target_slices)

    # Loop over all folds
    # TODO: Only do one fold in each script by giving it as argument
    # TODO: Save model with fold identifier
    for val_fold in range(len(folds)):
        train_patient, val_patient = get_train_and_val(folds, val_fold)
        train_df = label_df[label_df["patient_nr"].isin(train_patient)]
        val_df = label_df[label_df["patient_nr"].isin(val_patient)]

        train_patients.append(train_patient)
        val_patients.append(val_patient)
        train_dfs.append(train_df)
        val_dfs.append(val_df)

        # Set train/validation loaders
        train_set = PatientDataset(train_df, train_patient, args.target_slices, args.data_dir, DEVICE)
        val_set = PatientDataset(val_df, val_patient, args.target_slices, args.data_dir, DEVICE)

        train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

        # Initialise W&B settings
        wandb.init(project="braintriage")
        wandb.config.update({"model_type": args.name, "epochs": args.epochs, "batch_size": args.batch_size,
                             "learning_rate": args.learning_rate,
                             "n_features": args.n_features, "target_slices": args.target_slices,
                             "is_target_tuple": args.is_target_tuple,
                             "train_percentage": args.train_percentage})
        wandb.watch(combined_net)
        trainer = Trainer(model=combined_net, criterion=criterion, optimizer=optimizer, device=DEVICE,
                          train_loader=train_loader, val_loader=val_loader, n_epochs=args.epochs, model_dir=args.model_dir)
        trainer.train_and_validate()