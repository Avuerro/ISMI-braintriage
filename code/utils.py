import random
import numpy as np
import torch

def set_seed(seed):
    """
        Retrieved from https://discuss.pytorch.org/t/how-to-get-deterministic-behavior/18177/17
    """
    torch.backends.cudnn.deterministic = True
    # If size of newtork input is constant, benchmark might improve performance (causes only minor non-deterministic fluctuations)
    # torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)