import os
os.system("python3 -m wandb.cli login 8d7601a3f5545dac156785dbc02523182dcf0458")

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
sys.path.append(".")
sys.path.append("..")
from dataset.slice_dataset import SliceDataset
from models.omnipotent_resnet import Net
from train import Trainer
from utils import set_seed


### DEFAULT PARAMETERS ###
### Data parameters ###
DATA_DIR = '../../../data_sliced/train'
DS_DIR = '../../../data_split'
TARGET_SLICES = (0,32)                                   # The slices we will train on for each patient
FLIP_PROB = 0.5                                          # Probability of augmenting training data by flipping slices left to right
ROTATE_PROB = 0.5                                        # Probability of augmenting training data by randomly slightly rotating slices
### Model parameters ###
MODEL_DIR = '../models'                                  # Directory where best models are saved
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Train on GPU or CPU
N_FEATURES = 128                                         # The length of feature vectors that the CNN outputs/LSTM will use
### Train parameters ###
EPOCHS = 30
BATCH_SIZE = 16
LR = 0.0001
SEED = 420


### Argument parser ###
parser = argparse.ArgumentParser(description='Train a specified ResNet model.')
parser.add_argument('name', type=str, help="Name of the model")
parser.add_argument('resnet', type=str, help = "Type of ResNet to use (resnet18, resnet34 or resnet50)")
parser.add_argument('-s', type=int, nargs='?', dest="seed",
                    default = SEED, help="Seed for all random generators")
parser.add_argument('-d', type=str, nargs='?', dest="data_dir",
                    default = DATA_DIR, help="Path to directory with data")
parser.add_argument('-ds', type=str, nargs='?', dest="ds_dir",
                    default = DS_DIR, help="Path to directory with data split .csv files")                    
parser.add_argument('-lr', type=float, nargs='?', dest="learning_rate",
                    default = LR, help='Learning rate')
parser.add_argument('-e', type=int, nargs='?', dest="epochs",
                    default = EPOCHS, help='Number of epochs')
parser.add_argument('-b', type=int, nargs='?', dest="batch_size",
                    default = BATCH_SIZE, help="Batch size")
parser.add_argument('-m', type=str, nargs='?', dest="model_dir",
                    default = MODEL_DIR, help="Where models will be saved")
parser.add_argument('-f', type=int, nargs='?', dest="n_features",
                    default = N_FEATURES, help="Number of output features of last FC layer")
parser.add_argument('-ts', nargs='+', dest='target_slices', type=int,
                    default = TARGET_SLICES, help="Which slices to use for training")
parser.add_argument('-afp', nargs='?', dest='flip_prob', type=float,
                    default = FLIP_PROB, help="Probability of augmenting training data by flipping slices left to right")
parser.add_argument('-arp', nargs='?', dest='rotate_prob', type=float,
                    default = ROTATE_PROB, help="Probability of augmenting training data by randomly slightly rotating slices")
parser.add_argument('--tuple', action="store_true", dest="is_target_tuple",
                    help="Whether slices argument is tuple or not")
parser.add_argument('--pretrained', action="store_true", help="Whether networks are pretrained")


if __name__ == "__main__":
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Load and check data
    train_df = pd.read_csv(os.path.join(args.ds_dir, "train_df.csv"), names=["patient_nr", "slice_nr", "class"] ).sample(frac=1).reset_index(drop=True)
    print(f"\nNumber of unique patient numbers in training set: {len(np.unique(train_df['patient_nr']))}")
    print(f"Number of unique slice numbers in training set:   {len(np.unique(train_df['slice_nr']))}")
    print(f"Number of unique class values in training set:    {len(np.unique(train_df['class']))}")
    val_df = pd.read_csv(os.path.join(args.ds_dir, "val_df.csv"),  names=["patient_nr", "slice_nr", "class"]).sample(frac=1).reset_index(drop=True)
    print(f"\nNumber of unique patient numbers in validation set: {len(np.unique(val_df['patient_nr']))}")
    print(f"Number of unique slice numbers in validation set:   {len(np.unique(val_df['slice_nr']))}")
    print(f"Number of unique class values in validation set:    {len(np.unique(val_df['class']))}")

    # Load in correct model
    if args.resnet == "resnet50":
        model = models.resnet50(pretrained=args.pretrained)
    elif args.resnet == "resnet34":
        model = models.resnet34(pretrained=args.pretrained)
    elif args.resnet == "resnet18":
        model = models.resnet18(pretrained=args.pretrained)
    else:
        print(f'No resnet with name {args.resnet}')
        exit()
    # Change the Pre-Trained Model to our own Defined Model
    model = Net(model, args.name, args.n_features)

    ### Loss and optimizer ###
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    ### Create data generator ###
    # Set correct target slices
    if args.is_target_tuple:
        args.target_slices = tuple(args.target_slices)

    # Set train/validation loaders and train    
    train_set = SliceDataset(train_df, args.target_slices, args.data_dir, args.flip_prob, args.rotate_prob) # Train dataset should perform augmentation
    val_set = SliceDataset(val_df, args.target_slices, args.data_dir) # Validation dataset should not perform augmentation

    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # Initialise W&B settings
    train_percentage = float(len(np.unique(train_df['patient_nr']))) / float(len(np.unique(train_df['patient_nr'])) + len(np.unique(val_df['patient_nr'])))
    wandb.init(project="braintriage", entity="angry-chickens")
    wandb.config.update({"model_type":args.name, "epochs":args.epochs, "batch_size":args.batch_size, "learning_rate":args.learning_rate,
                         "n_features":args.n_features, "target_slices":args.target_slices, "is_target_tuple":args.is_target_tuple,
                         "train_percentage":train_percentage})
    wandb.watch(model)
    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, device=DEVICE,
                    train_loader=train_loader, val_loader=val_loader, n_epochs=args.epochs, model_dir = args.model_dir, verbose=True)
    trainer.train_and_validate()