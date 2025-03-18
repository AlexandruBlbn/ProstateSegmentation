import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import get_dataloaders
from U_net import UNet

def dice_score(pred, target, num_classes, smooth=1e-5):
    pred_one_hot = F.one_hot(pred, num_classes=num_classes).permute(0,3,1,2).float()
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0,3,1,2).float()
    dice = []
    for i in range(num_classes):
        p = pred_one_hot[:, i]
        t = target_one_hot[:, i]
        intersection = (p * t).sum(dim=(1,2))
        union = p.sum(dim=(1,2)) + t.sum(dim=(1,2))
        dice.append(((2 * intersection + smooth) / (union + smooth)).mean().item())
    return dice, sum(dice)/len(dice)

def train_epoch(model, device, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)

def validate_epoch(model, device, dataloader, num_classes):
    model.eval()
    dice_per_class = [0.0] * num_classes
    mean_dice = 0.0
    count = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            dice, mean = dice_score(preds, labels, num_classes)
            dice_per_class = [d + dc for d, dc in zip(dice, dice_per_class)]
            mean_dice += mean
            count += 1
    dice_per_class = [d / count for d in dice_per_class]
    mean_dice = mean_dice / count
    return dice_per_class, mean_dice

def train_model(json_path, epochs=10, batch_size=4, lr=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu'):
    train_loader, val_loader = get_dataloaders(json_path, batch_size=batch_size, augment=True)
    model = UNet(n_channels=3, n_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        train_loss = train_epoch(model, device, train_loader, optimizer, criterion)
        dice_per_class, mean_dice = validate_epoch(model, device, val_loader, num_classes=3)
        print(f"Epoch {epoch+1}/{epochs} Loss: {train_loss:.4f} Dice: {dice_per_class} Mean Dice: {mean_dice:.4f}")
    return model

if __name__ == '__main__':
    train_model("train.json", epochs=10, batch_size=4)
