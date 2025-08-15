import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import get_dataloaders
from U_net import UNet

def dice_score(pred, target, num_classes, smooth=1e-5):
    """
    Calculează Dice Score pentru evaluarea segmentării.
    
    Dice Score măsoară overlap-ul dintre predicție și ground truth.
    Formula: Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Args:
        pred (Tensor): predicțiile modelului cu shape (batch_size, H, W)
        target (Tensor): ground truth cu shape (batch_size, H, W)
        num_classes (int): numărul de clase
        smooth (float): factor de regularizare pentru evitarea diviziunii cu 0
    
    Returns:
        tuple: (dice_per_class, mean_dice)
            - dice_per_class: lista cu Dice Score pentru fiecare clasă
            - mean_dice: media Dice Score-urilor pentru toate clasele
    """
    # Convertește predicțiile și target-urile la format one-hot
    pred_one_hot = F.one_hot(pred, num_classes=num_classes).permute(0,3,1,2).float()
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0,3,1,2).float()
    
    dice = []
    # Calculează Dice Score pentru fiecare clasă
    for i in range(num_classes):
        p = pred_one_hot[:, i]  # Predicțiile pentru clasa i
        t = target_one_hot[:, i]  # Ground truth pentru clasa i
        
        # Calculează intersecția și uniunea
        intersection = (p * t).sum(dim=(1,2))
        union = p.sum(dim=(1,2)) + t.sum(dim=(1,2))
        
        # Calculează Dice Score cu factor de regularizare
        dice_class = ((2 * intersection + smooth) / (union + smooth)).mean().item()
        dice.append(dice_class)
    
    # Returnează Dice Score-urile per clasă și media
    return dice, sum(dice)/len(dice)

def train_epoch(model, device, dataloader, optimizer, criterion):
    """
    Antrenează modelul pentru o epocă.
    
    Args:
        model: modelul U-Net
        device: device-ul de calcul ('cpu' sau 'cuda')
        dataloader: DataLoader pentru datele de antrenare
        optimizer: optimizatorul (ex: Adam)
        criterion: funcția de loss (ex: CrossEntropyLoss)
    
    Returns:
        float: loss-ul mediu pentru epoca curentă
    """
    model.train()  # Setează modelul în modul de antrenare
    running_loss = 0.0
    
    for images, labels in dataloader:
        # Transferă datele pe device-ul de calcul
        images = images.to(device)
        labels = labels.to(device)
        
        # Resetează gradientii
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculează loss-ul
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Actualizează parametrii
        optimizer.step()
        
        # Acumulează loss-ul
        running_loss += loss.item() * images.size(0)
    
    # Returnează loss-ul mediu pe epocă
    return running_loss / len(dataloader.dataset)

def validate_epoch(model, device, dataloader, num_classes):
    """
    Evaluează modelul pe setul de validare.
    
    Args:
        model: modelul U-Net
        device: device-ul de calcul
        dataloader: DataLoader pentru datele de validare
        num_classes: numărul de clase pentru segmentare
    
    Returns:
        tuple: (dice_per_class, mean_dice)
            - dice_per_class: Dice Score pentru fiecare clasă
            - mean_dice: media Dice Score-urilor
    """
    model.eval()  # Setează modelul în modul de evaluare
    dice_per_class = [0.0] * num_classes
    mean_dice = 0.0
    count = 0
    
    with torch.no_grad():  # Dezactivează calculul gradientilor pentru eficiență
        for images, labels in dataloader:
            # Transferă datele pe device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Convertește logits la predicții de clase
            preds = torch.argmax(outputs, dim=1)
            
            # Calculează Dice Score
            dice, mean = dice_score(preds, labels, num_classes)
            
            # Acumulează rezultatele
            dice_per_class = [d + dc for d, dc in zip(dice, dice_per_class)]
            mean_dice += mean
            count += 1
    
    # Calculează media pe toate batch-urile
    dice_per_class = [d / count for d in dice_per_class]
    mean_dice = mean_dice / count
    
    return dice_per_class, mean_dice

def train_model(json_path, epochs=10, batch_size=4, lr=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Funcția principală pentru antrenarea modelului U-Net.
    
    Args:
        json_path (str): calea către fișierul JSON cu datele
        epochs (int): numărul de epoci de antrenare
        batch_size (int): dimensiunea batch-ului
        lr (float): learning rate pentru optimizer
        device (str): device-ul de calcul ('cpu' sau 'cuda')
    
    Returns:
        model: modelul antrenat
    """
    print(f"🚀 Începe antrenarea pe device: {device}")
    print(f"⚙️  Parametri: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    
    # Creează DataLoader-ele pentru antrenare și validare
    train_loader, val_loader = get_dataloaders(json_path, batch_size=batch_size, augment=True)
    print(f"📊 Dataset încărcat: {len(train_loader.dataset)} samples pentru antrenare, "
          f"{len(val_loader.dataset)} pentru validare")
    
    # Creează modelul U-Net (3 canale input, 3 clase output)
    model = UNet(n_channels=3, n_classes=3).to(device)
    print(f"🧠 Model creat cu {sum(p.numel() for p in model.parameters())} parametri")
    
    # Configurează optimizatorul și funcția de loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print("🏋️  Începe antrenarea...")
    print("-" * 80)
    
    # Bucla principală de antrenare
    for epoch in range(epochs):
        # Antrenează o epocă
        train_loss = train_epoch(model, device, train_loader, optimizer, criterion)
        
        # Evaluează pe setul de validare
        dice_per_class, mean_dice = validate_epoch(model, device, val_loader, num_classes=3)
        
        # Afișează rezultatele
        class_names = ["Fundal", "Prostata", "Zona periferică"]
        dice_str = " | ".join([f"{name}: {dice:.4f}" for name, dice in zip(class_names, dice_per_class)])
        
        print(f"Epoca {epoch+1:2d}/{epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Mean Dice: {mean_dice:.4f}")
        print(f"            Dice per clasă: {dice_str}")
        print("-" * 80)
    
    print("✅ Antrenarea completă!")
    return model

if __name__ == '__main__':
    # Exemplu de utilizare
    print("🔬 Sistem de Segmentare a Prostatei cu U-Net")
    print("=" * 50)
    
    # Verifică dacă există un fișier JSON de test
    import os
    if os.path.exists("train.json"):
        print("📁 Fișierul train.json găsit. Începe antrenarea...")
        train_model("train.json", epochs=10, batch_size=4)
    else:
        print("⚠️  Fișierul train.json nu a fost găsit!")
        print("📝 Creează un fișier train.json cu următoarea structură:")
        print("""
{
  "Pacient": [
    {
      "data": "path/to/image1.png",
      "label": "path/to/mask1.png"
    },
    {
      "data": "path/to/image2.png", 
      "label": "path/to/mask2.png"
    }
  ]
}
        """)
        print("\n💡 Sau rulează demo.py pentru o demonstrație a sistemului!")
        print("   python demo.py")
