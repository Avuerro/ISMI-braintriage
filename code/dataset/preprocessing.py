import numpy as np
import pandas as pd
import os
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

DATA_DIR = "../../data/train/"

def preprocess(X):
    
    '''
    X = torch.load(...)'s output (hence, a Torch Tensor)
    '''
    # Define transformations:
    
    preprocessing = transforms.Compose([
        transforms.CenterCrop(425),
        transforms.ToTensor()
    ])
    # Transform Torch Tensor to NP Array
    X_array = X.numpy()
    
    # Scale Data
    X_img = (X_array / np.max(X_array) * 255).astype('uint8')
    
    pil_img_T1 = Image.fromarray(X_img[0,:])
    pil_img_T2 = Image.fromarray(X_img[1,:])
    pil_img_T2_FLAIR = Image.fromarray(X_img[2,:])
    
    T1 = torch.squeeze(preprocessing(pil_img_T1),0)
    T2 = torch.squeeze(preprocessing(pil_img_T2),0)
    T2_FLAIR = torch.squeeze(preprocessing(pil_img_T2_FLAIR),0)
    
    preprocessed_slice = torch.stack([T1, T2, T2_FLAIR], dim=0)
    
    return preprocessed_slice


def get_dataset_mean_std():
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
    from .slice_dataset import SliceDataset # Might cause issue due to circular dependency with this file
    label_df = pd.read_csv(os.path.join(DATA_DIR, "labels_slices.csv"), names=["patient_nr", "slice_nr", "class"])
    dataloader = data.DataLoader(SliceDataset(label_df, (0,31), DATA_DIR, False), batch_size=1000, num_workers=0)

    mean, std, total_samples = 0., 0., 0.
    for batch,_ in tqdm(dataloader):
        n_batch_samples = batch.size(0)
        batch = batch.view(n_batch_samples, batch.size(1), -1)
        mean += batch.mean(2).sum(0)
        std += batch.std(2).sum(0)
        total_samples += n_batch_samples
    print(total_samples)

    return mean / total_samples, std / total_samples

if __name__ == "__main__":
    print(get_dataset_mean_std())