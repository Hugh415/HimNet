import warnings
warnings.filterwarnings("ignore", message="It is not recommended to directly access")
import numpy as np
import torch
import secrets
import random


def generate_random_seed():
    return secrets.randbelow(2 ** 32)

def set_seed(seed):
    print(f"[INFO] Using random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False