from dataset.slice_dataframes import get_slice_train_val_dataframes
from dataset.slice_dataset import SliceDataset
from models.omnipotent_resnet import Net
from torchvision import models
from tqdm.notebook import tqdm

import pandas as pd
import torch
from torch.utils import data
import os


TRAIN_PERCENTAGE = 0.9
DATA_DIR = "D:/data/train"
CNN_DIR = "../resnet34_009.pt"
MODEL_NAME = "resnet34"
PRETRAINED = False
N_FEATURES = 128
TARGET_SLICES = (0,32)
BATCH_SIZE = 64


#if __name__ == "__main__":
# Load in correct model
if MODEL_NAME == "resnet50":
    model = models.resnet50(pretrained=PRETRAINED)
elif MODEL_NAME == "resnet34":
    model = models.resnet34(pretrained=PRETRAINED)
elif MODEL_NAME == "resnet18":
    model = models.resnet18(pretrained=PRETRAINED)
else:
    print(f'No model with name {MODEL_NAME}')
    exit()
# Change the Pre-Trained Model to our own Defined Model
model = Net(model, MODEL_NAME, N_FEATURES)
model.load_state_dict(torch.load(CNN_DIR))

# Set dataloader
label_df = pd.read_csv(os.path.join(DATA_DIR,"labels_slices.csv"), names = ["patient_nr", "slice_nr", "class"])
failure_set = SliceDataset(label_df, TARGET_SLICES, DATA_DIR)
failure_loader = data.DataLoader(failure_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count())

model.eval()
all_outputs = []
for images, _ in tqdm(failure_loader):
    outputs = model(images).detach()
    all_outputs.extend(outputs)
    
    
    
    