### Import packages ###
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import models
import pandas as pd
import numpy as np
import os, argparse

### Local imports ###
from dataset.slice_dataframes import get_slice_train_val_dataframes
from dataset.slice_dataset import SliceDataset
from models.omnipotent_resnet import Net
from train.train import Trainer


### DEFAULT PARAMETERS ###
### Data parameters ###
DATA_DIR = '/Users/fbergh/Documents/Radboud/master/1/ISMI/project/ISMI-braintriage/data/BrainTriage/sliced_data'
TARGET_SLICES = (0,31)                                   # The slices we will train on for each patient
TRAIN_PERCENTAGE = 0.9                                   # Percentage of data that will be used for training
### Model parameters ###
MODEL_DIR = '../models'                                  # Directory where best models are saved
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Train on GPU or CPU
N_FEATURES = 128                                         # The length of feature vectors that the CNN outputs/LSTM will use
### Train parameters ###
EPOCHS = 50
BATCH_SIZE = 16


### Argument parser ###
parser = argparse.ArgumentParser(description='Train a specified ResNet model.')
parser.add_argument('name', type=str, help="Name of the pre-trained model")
parser.add_argument('-e', type=int, nargs='?', dest="epochs",
                    default = EPOCHS, help='Number of epochs')
parser.add_argument('-b', type=int, nargs='?', dest="batch_size",
                    default = BATCH_SIZE, help="Batch size")
parser.add_argument('-m', type=str, nargs='?', dest="model_dir",
                    default = MODEL_DIR, help="Where models will be saved")
parser.add_argument('-f', type=int, nargs='?', dest="n_features",
                    default = N_FEATURES, help="Number of output features of last FC layer")
parser.add_argument('-s', nargs='+', dest='target_slices',
                    default = TARGET_SLICES, help="Which slices to use for training")
parser.add_argument('--tuple', action="store_true", dest="is_target_tuple",
                    help="Whether slices argument is tuple or not")



if __name__ == "__main__":
    args = parser.parse_args()

    # Load and check data
    label_df = pd.read_csv(os.path.join(DATA_DIR,"labels_slices.csv"), names = ["patient_nr", "slice_nr", "class"])
    label_df["class"] = label_df["class"].astype("int8")
    patient_list = np.unique(label_df["patient_nr"])
    print(label_df.head(), f"Dataframe shape: {label_df.shape}", sep="\n")
    print(f"\nNumber of unique patient numbers: {len(np.unique(label_df['patient_nr']))}")
    print(f"Number of unique slice numbers:   {len(np.unique(label_df['slice_nr']))}")
    print(f"Number of unique class values:    {len(np.unique(label_df['class']))}")

    # Load in correct model
    if args.name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif args.name == "resnet34":
        model = models.resnet34(pretrained=True)
    elif args.name == "resnet18":
        model = models.resnet18(pretrained=True)
    else:
        print(f'No model with name {args.name}')
        exit()
    # Change the Pre-Trained Model to our own Defined Model
    model = Net(model, args.name, args.n_features)

    ### Loss and optimizer ###
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)

    ### Create data generator ###
    train_df, val_df = get_slice_train_val_dataframes(label_df, train_percentage = TRAIN_PERCENTAGE)
    
    # Set correct target slices
    if args.is_target_tuple:
        args.target_slices = tuple(args.target_slices)

    # Set train/validation loaders and train    
    train_set = SliceDataset(train_df, args.target_slices, DATA_DIR)
    val_set = SliceDataset(val_df, args.target_slices, DATA_DIR)

    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle = True)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle = False)

    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, 
                    train_loader=train_loader, val_loader=val_loader, n_epochs=args.epochs, model_dir = args.model_dir)
    trainer.train_and_validate()