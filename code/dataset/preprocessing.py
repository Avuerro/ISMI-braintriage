import numpy as np
import pandas as pd
import os
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

DATA_DIR = "../../data/train/"
# For convenience convert to tensor so it's ready for preprocess()
MEANS = torch.Tensor([96.3782, 106.0149,  58.2791])
STDS = torch.Tensor([163.7764, 172.2486,  94.4285])
IMG_SIZE = 512

def preprocess(X, center_crop_target = 425):
    '''
    X = PyTorch Tensor
    '''

    # Convert to (512,512,3) for standardising and convert back
    X = X.view(IMG_SIZE, IMG_SIZE, 3)
    X_standard = (X - MEANS) / STDS
    X_standard = X_standard.view(3, IMG_SIZE, IMG_SIZE)

    # Center-crop image manually (PIL does not like floats)
    crop_idx = (IMG_SIZE - center_crop_target)//2
    X_cropped = X_standard[:, crop_idx:-crop_idx, crop_idx:-crop_idx]
    return X_cropped

def get_dataset_mean_std(dataloader):
    '''
        Retrieves the mean and standard deviation per acquisition over the training set
        Returns:
        MEAN
             T1       T2         T2-Flair
            [96.3782, 106.0149,  58.2791]
        STANDARD DEVIATION
             T1        T2         T2-Flair
            [163.7764, 172.2486,  94.4285]
    '''
    mean, std, total_samples = 0., 0., 0.
    # print(dataloader[0].shape)
    for batch,_ in tqdm(dataloader):
        n_batch_samples = batch.size(0)
        batch = batch.view(n_batch_samples, batch.size(1), -1)
        mean += batch.mean(2).sum(0)        
        std += batch.std(2).sum(0)
        total_samples += n_batch_samples

    return mean / total_samples, std / total_samples

if __name__ == "__main__":
    # X = torch.load(os.path.join(DATA_DIR, f"3_1.pt"))
    # print(X.shape)
    # out = preprocess(X)
    # print(out.mean(), out.min(), out.max(), out.shape)
    # import matplotlib.pyplot as plt
    # ax = plt.subplot(1, 3, 1)
    # ax.imshow(out[0])
    # ax = plt.subplot(1, 3, 2)
    # ax.imshow(out[1])
    # ax = plt.subplot(1, 3, 3)
    # ax.imshow(out[2])
    # plt.show()
    get_dataset_mean_std()