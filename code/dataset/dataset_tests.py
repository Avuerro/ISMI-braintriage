import numpy as np
import pandas as pd
import os
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from slice_dataset import SliceDataset
from preprocessing import get_dataset_mean_std
# from . import preprocess,get_dataset_mean_std


"""
    This file is usefull for testing some classes and methods in the dataset module
"""



DATA_DIR = "../../../data_sliced"
MEANS = [96.3782, 106.0149,  58.2791]
STDS = [163.7764, 172.2486,  94.4285]
IMG_SIZE = 512

label_df = pd.read_csv(os.path.join(DATA_DIR, "labels_slices.csv"), names=["patient_nr", "slice_nr", "class"])
dataloader = data.DataLoader(SliceDataset(label_df, (0,32), DATA_DIR, False), batch_size=1000, num_workers=0)




if __name__ == "__main__":
    # print(os.listdir('../../../data_sliced'))
    # X = torch.load(os.path.join(DATA_DIR, f"3_1.pt"))
    # print(X.shape)
    # out = preprocess(X)
    # print(out.min(), out.max(), out.shape)
    # import matplotlib.pyplot as plt
    # ax = plt.subplot(1, 3, 1)
    # ax.imshow(out[0], cmap="gray")
    # ax = plt.subplot(1, 3, 2)
    # ax.imshow(out[1], cmap="gray")
    # ax = plt.subplot(1, 3, 3)
    # ax.imshow(out[2], cmap="gray")
    # plt.show()


    mean_divided, std_divided = get_dataset_mean_std(label_df,dataloader)
    print(mean_divided)
    print(std_divided)
    # print(dataloader[0][0].shape)
    print(mean_divided.shape)
    print(mean_divided[0])
    print(dataloader)
    # print(normalized)
    for i in tqdm(dataloader):
        print(i[0].shape)
        normalized = (i[0] - mean_divided) / std_divided
        print(normalized.shape)
        print(normalized.mean())
        break