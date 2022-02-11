import os
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T


def get_dataloaders(miniset_dataroot, valid_dataroot, batch_size):
    data_transforms = {
        'train': T.Compose([
            T.RandomResizedCrop(256),
            T.RandomHorizontalFlip(),
            T.ColorJitter(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': T.Compose([
            T.Resize(300),
            T.CenterCrop(256),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    miniset_dataset = datasets.ImageFolder(miniset_dataroot, data_transforms['train'])
    valid_dataset = datasets.ImageFolder(valid_dataroot, data_transforms['valid'])
    
    train_dataloader = DataLoader(miniset_dataset, shuffle=True, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size)
    
    return train_dataloader, valid_dataloader
