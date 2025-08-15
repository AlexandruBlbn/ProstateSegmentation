"""
Demo script pentru sistemul de segmentare a prostatei.
Acest script demonstrează cum să folosești componenetele sistemului.
"""

import torch
import torch.nn.functional as F
from U_net import UNet
from dataloader import PatientDataset, get_dataloaders
import json
import os
from PIL import Image
import numpy as np

def demo_model_architecture():
    """Demonstrează arhitectura modelului U-Net."""
    print("=== Demonstrație Arhitectură U-Net ===")
    
    # Creează modelul
    model = UNet(n_channels=3, n_classes=3)
    print(f"Model creat cu {sum(p.numel() for p in model.parameters())} parametri")
    
    # Test cu o imagine simulată
    dummy_input = torch.randn(1, 3, 256, 256)  # Batch=1, RGB, 256x256
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        
        # Convertește la predicții de clasă
        predictions = torch.argmax(output, dim=1)
        print(f"Predictions shape: {predictions.shape}")
        print(f"Clase prezise unice: {torch.unique(predictions).tolist()}")
    
    print("✓ Arhitectura modelului funcționează corect!\n")

def demo_dataloader():
    """Demonstrează funcționarea dataloader-ului."""
    print("=== Demonstrație DataLoader ===")
    
    # Creează un JSON de test
    demo_json_path = "/tmp/demo_data.json"
    demo_data = {
        "Pacient": [
            {
                "data": "/tmp/demo_image.png",
                "label": "/tmp/demo_label.png"
            }
        ]
    }
    
    # Creează imagini demo
    os.makedirs("/tmp", exist_ok=True)
    
    # Imagine RGB demo (256x256)
    demo_image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    demo_image.save("/tmp/demo_image.png")
    
    # Mască demo (256x256) cu valori 0, 1, 2
    demo_mask = Image.fromarray(np.random.randint(0, 3, (256, 256), dtype=np.uint8))
    demo_mask.save("/tmp/demo_label.png")
    
    # Salvează JSON-ul demo
    with open(demo_json_path, 'w') as f:
        json.dump(demo_data, f)
    
    try:
        # Testează dataset-ul
        dataset = PatientDataset(demo_json_path, augment=False)
        print(f"Dataset size: {len(dataset)}")
        
        # Obține un sample
        image, label = dataset[0]
        print(f"Image shape: {image.shape}, dtype: {image.dtype}")
        print(f"Label shape: {label.shape}, dtype: {label.dtype}")
        print(f"Label unique values: {torch.unique(label).tolist()}")
        
        print("✓ DataLoader funcționează corect!")
        
    except Exception as e:
        print(f"⚠ Nu s-a putut testa dataloader-ul: {e}")
    
    finally:
        # Curăță fișierele demo
        for file in [demo_json_path, "/tmp/demo_image.png", "/tmp/demo_label.png"]:
            if os.path.exists(file):
                os.remove(file)
    
    print()

def demo_training_functions():
    """Demonstrează funcțiile de antrenare."""
    print("=== Demonstrație Funcții de Antrenare ===")
    
    # Import funcții de antrenare
    from train import dice_score
    
    # Creează predicții și targets demo
    num_classes = 3
    batch_size = 2
    height, width = 64, 64
    
    # Predicții simulate (logits)
    pred_logits = torch.randn(batch_size, num_classes, height, width)
    predictions = torch.argmax(pred_logits, dim=1)
    
    # Target-uri simulate
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Calculează Dice Score
    dice_per_class, mean_dice = dice_score(predictions, targets, num_classes)
    
    print(f"Dice Score per clasă: {[f'{d:.4f}' for d in dice_per_class]}")
    print(f"Mean Dice Score: {mean_dice:.4f}")
    
    print("✓ Funcțiile de antrenare funcționează corect!\n")

def demo_inference():
    """Demonstrează procesul de inferență."""
    print("=== Demonstrație Inferență ===")
    
    # Creează modelul și setează în modul de evaluare
    model = UNet(n_channels=3, n_classes=3)
    model.eval()
    
    # Imagine demo
    demo_image = torch.randn(1, 3, 256, 256)
    
    print(f"Input image shape: {demo_image.shape}")
    
    # Inferență
    with torch.no_grad():
        # Forward pass
        logits = model(demo_image)
        
        # Convertește la probabilități
        probabilities = F.softmax(logits, dim=1)
        
        # Predicții finale
        predictions = torch.argmax(probabilities, dim=1)
        
        print(f"Output logits shape: {logits.shape}")
        print(f"Predictions shape: {predictions.shape}")
        
        # Analizează predicțiile
        unique_classes, counts = torch.unique(predictions, return_counts=True)
        print(f"Clase prezise:")
        for cls, count in zip(unique_classes.tolist(), counts.tolist()):
            percentage = (count / predictions.numel()) * 100
            class_names = ["Fundal", "Prostata", "Zona periferică"]
            print(f"  - Clasa {cls} ({class_names[cls]}): {count} pixeli ({percentage:.1f}%)")
    
    print("✓ Inferența funcționează corect!\n")

def main():
    """Funcția principală care rulează toate demonstrațiile."""
    print("🔬 DEMO: Sistem de Segmentare a Prostatei cu U-Net")
    print("=" * 55)
    print()
    
    # Verifică disponibilitatea CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device folosit: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()
    
    # Rulează demonstrațiile
    demo_model_architecture()
    demo_dataloader()
    demo_training_functions()
    demo_inference()
    
    print("🎉 Toate demonstrațiile au fost completate cu succes!")
    print("\nPentru a antrena modelul pe datele tale:")
    print("1. Pregătește dataset-ul în formatul JSON specificat")
    print("2. Rulează: python train.py")
    print("3. Sau folosește: from train import train_model; train_model('path/to/data.json')")

if __name__ == "__main__":
    main()