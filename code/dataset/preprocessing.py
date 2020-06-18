import numpy as np
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
    
    pil_img = Image.fromarray(X_img)
    
    return preprocessing(pil_img)