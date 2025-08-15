import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from PIL import Image

class PatientDataset(Dataset):
    """
    Dataset pentru imagini medicale și măștile de segmentare corespunzătoare.
    
    Acest dataset încarcă imagini RGB și măștile de segmentare din fișiere,
    aplicând transformări de augmentare dacă este specificat.
    
    Args:
        json_path (str): calea către fișierul JSON cu datele
        augment (bool): dacă să aplice augmentări de date
    
    Formatul JSON așteptat:
    {
        "Pacient": [
            {
                "data": "path/to/image.png",
                "label": "path/to/mask.png"
            }
        ]
    }
    """
    def __init__(self, json_path, augment=False):
        # Încarcă datele din JSON
        with open(json_path, 'r') as f:
            self.data = json.load(f)["Pacient"]
        
        # Transformări de bază (conversie la tensor)
        base_transforms = [T.ToTensor()]
        
        if augment:
            # Augmentări pentru îmbunătățirea generalizării modelului
            augment_transforms = [
                T.RandomHorizontalFlip(),           # Flip orizontal aleator
                T.RandomRotation(degrees=30)        # Rotație aleatoare ±30°
            ]
            # Aplică augmentările atât la imagine cât și la mască
            self.transform = T.Compose(augment_transforms + base_transforms)
            self.target_transform = T.Compose(augment_transforms + base_transforms)
        else:
            # Doar transformările de bază
            self.transform = T.Compose(base_transforms)
            self.target_transform = T.Compose(base_transforms)
    
    def __len__(self):
        """Returnează numărul de samples din dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returnează un sample (imagine, mască) la indexul specificat.
        
        Args:
            idx (int): indexul sample-ului
            
        Returns:
            tuple: (image_tensor, label_tensor)
                - image_tensor: tensor RGB cu shape (3, H, W)
                - label_tensor: tensor cu clasele cu shape (H, W)
        """
        item = self.data[idx]
        
        # Încarcă imaginea ca RGB
        image = Image.open(item["data"]).convert('RGB')
        # Încarcă masca ca grayscale (1 canal)
        label = Image.open(item["label"]).convert('L')
        
        # Aplică transformările
        image = self.transform(image)
        label = self.target_transform(label)
        
        # Convertește label-ul la format de clase întregi
        # Înmulțește cu 255 și convertește la long pentru CrossEntropyLoss
        label = (label * 255).long().squeeze(0)
        
        return image, label

def get_dataloaders(json_path, batch_size=4, num_workers=4, augment=True, val_split=0.2):
    """
    Creează DataLoader-ele pentru antrenare și validare.
    
    Această funcție împarte dataset-ul în seturi de antrenare și validare,
    aplicând augmentări doar pe setul de antrenare.
    
    Args:
        json_path (str): calea către fișierul JSON cu datele
        batch_size (int): dimensiunea batch-ului
        num_workers (int): numărul de worker-i pentru încărcarea datelor
        augment (bool): dacă să aplice augmentări pe setul de antrenare
        val_split (float): procentul de date pentru validare (0.0-1.0)
    
    Returns:
        tuple: (train_loader, val_loader)
            - train_loader: DataLoader pentru antrenare
            - val_loader: DataLoader pentru validare
    """
    # Creează dataset-ul cu augmentări pentru antrenare
    dataset = PatientDataset(json_path, augment=augment)
    
    # Calculează dimensiunile seturilor de date
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    
    # Împarte dataset-ul aleator în antrenare și validare
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Creează DataLoader-ele
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,           # Amestecă datele pentru antrenare
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,          # Nu amesteca datele pentru validare
        num_workers=num_workers
    )
    
    return train_loader, val_loader
