import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

def imagenet_val_loader(val_path, batch_size=2, input_size=256, center=224):
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(center),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(val_path, transform=transform)
    g = torch.Generator()
    g.manual_seed(0)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers=4, pin_memory=True, generator=g)

