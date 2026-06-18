import torch
import random
import numpy as np
from torchvision import transforms

def seed_eveything(is_cuda):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if is_cuda:
        torch.cuda.manual_seed(0)

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],     # ImageNet mean
    std=[0.229, 0.224, 0.225]       # ImageNet std
)