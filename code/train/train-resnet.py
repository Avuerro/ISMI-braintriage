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
from dataset.slice_dataset import SliceDataset
from models.omnipotent_resnet import Net
from train.train import Trainer
from utils import set_seed


### DEFAULT PARAMETERS ###
### Data parameters ###
DATA_DIR = '../../../sliced_data/train'
DS_DIR = '../../../data_split'
TARGET_SLICES = (0,32)                                   # The slices we will train on for each patient
TRAIN_PERCENTAGE = 0.9                                   # Percentage of data that will be used for training
### Model parameters ###
MODEL_DIR = '../models'                                  # Directory where best models are saved
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Train on GPU or CPU
N_FEATURES = 128                                         # The length of feature vectors that the CNN outputs/LSTM will use
### Train parameters ###
EPOCHS = 15
BATCH_SIZE = 64
LR = 0.0001
SEED = 420


### Argument parser ###
parser = argparse.ArgumentParser(description='Train a specified ResNet model.')
parser.add_argument('name', type=str, help="Name of the pre-trained model")
parser.add_argument('-s', type=int, nargs='?', dest="seed",
                    default = SEED, help="Seed for all random generators")
parser.add_argument('-d', type=str, nargs='?', dest="data_dir",
                    default = DATA_DIR, help="Path to directory with data")
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
parser.add_argument('-ts', nargs='+', dest='target_slices',
                    default = TARGET_SLICES, help="Which slices to use for training")
parser.add_argument('-tp', type=float, nargs='?', dest="train_percentage",
                    default = TRAIN_PERCENTAGE, help="Percentage of data to use for training")
parser.add_argument('--tuple', action="store_true", dest="is_target_tuple",
                    help="Whether slices argument is tuple or not")
parser.add_argument('--pretrained', action="store_true", help="Whether networks are pretrained")


if __name__ == "__main__":
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Load and check data
    label_df = pd.read_csv(os.path.join(args.data_dir,"labels_slices.csv"), names = ["patient_nr", "slice_nr", "class"])
    label_df["class"] = label_df["class"].astype("int8")
    print(label_df.head(), f"Dataframe shape: {label_df.shape}", sep="\n")
    print(f"\nNumber of unique patient numbers: {len(np.unique(label_df['patient_nr']))}")
    print(f"Number of unique slice numbers:   {len(np.unique(label_df['slice_nr']))}")
    print(f"Number of unique class values:    {len(np.unique(label_df['class']))}")

    # Load in correct model
    if args.name == "resnet50":
        model = models.resnet50(pretrained=args.pretrained)
    elif args.name == "resnet34":
        model = models.resnet34(pretrained=args.pretrained)
    elif args.name == "resnet18":
        model = models.resnet18(pretrained=args.pretrained)
    else:
        print(f'No model with name {args.name}')
        exit()
    # Change the Pre-Trained Model to our own Defined Model
    model = Net(model, args.name, args.n_features)

    ### Loss and optimizer ###
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    ### Create data generator ###
    train_df = pd.read_csv(os.path.join(DS_DIR, "train_df.csv"), index_col = 0).sample(frac=1).reset_index(drop=True)
    val_df = pd.read_csv(os.path.join(DS_DIR, "val_df.csv"), index_col = 0).sample(frac=1).reset_index(drop=True)
    
    
    # Set correct target slices
    if args.is_target_tuple:
        args.target_slices = tuple(args.target_slices)

    # Set train/validation loaders and train    
    train_set = SliceDataset(train_df, args.target_slices, args.data_dir)
    val_set = SliceDataset(val_df, args.target_slices, args.data_dir)

    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # Initialise W&B settings
    wandb.init(project="braintriage", entity="angry-chickens")
    wandb.config.update({"model_type":args.name, "epochs":args.epochs, "batch_size":args.batch_size, "learning_rate":args.learning_rate,
                         "n_features":args.n_features, "target_slices":args.target_slices, "is_target_tuple":args.is_target_tuple,
                         "train_percentage":args.train_percentage})
    wandb.watch(model)
    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, device=DEVICE,
                    train_loader=train_loader, val_loader=val_loader, n_epochs=args.epochs, model_dir = args.model_dir, verbose=True)
    trainer.train_and_validate()
