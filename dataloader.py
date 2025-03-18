import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from PIL import Image

class PatientDataset(Dataset):
    def __init__(self, json_path, augment=False):
        with open(json_path, 'r') as f:
            self.data = json.load(f)["Pacient"]
        base_transforms = [T.ToTensor()]
        if augment:
            augment_transforms = [T.RandomHorizontalFlip(), T.RandomRotation(degrees=30)]
            self.transform = T.Compose(augment_transforms + base_transforms)
            self.target_transform = T.Compose(augment_transforms + base_transforms)
        else:
            self.transform = T.Compose(base_transforms)
            self.target_transform = T.Compose(base_transforms)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["data"]).convert('RGB')
        label = Image.open(item["label"]).convert('L')
        image = self.transform(image)
        label = self.target_transform(label)
        label = (label * 255).long().squeeze(0)
        return image, label

def get_dataloaders(json_path, batch_size=4, num_workers=4, augment=True, val_split=0.2):
    dataset = PatientDataset(json_path, augment=augment)
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
