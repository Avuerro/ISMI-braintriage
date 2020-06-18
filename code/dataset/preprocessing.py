import numpy as np
import torch
from torchvision import transforms
from PIL import Image


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