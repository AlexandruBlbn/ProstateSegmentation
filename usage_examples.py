"""
Exemplu simplu de utilizare a sistemului de segmentare a prostatei.
Acest fișier demonstrează cum să folosești componentele principale.
"""

# Exemplu 1: Cum să creezi și folosești modelul U-Net
def example_model_usage():
    """Exemplu de creare și utilizare a modelului U-Net."""
    from U_net import UNet
    import torch
    
    # Creează modelul pentru imagini RGB (3 canale) și 3 clase
    model = UNet(n_channels=3, n_classes=3)
    
    # Exemplu de input: batch de 2 imagini 256x256 RGB
    dummy_input = torch.randn(2, 3, 256, 256)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
        predictions = torch.argmax(output, dim=1)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Predictions shape: {predictions.shape}")

# Exemplu 2: Cum să pregătești datele
def example_data_preparation():
    """Exemplu de pregătire a datelor pentru antrenare."""
    import json
    
    # Structura JSON pentru dataset
    data_structure = {
        "Pacient": [
            {
                "data": "images/patient_001.png",
                "label": "masks/patient_001_mask.png"
            },
            {
                "data": "images/patient_002.png", 
                "label": "masks/patient_002_mask.png"
            }
            # ... mai multe perechi de imagine-mască
        ]
    }
    
    # Salvează structura în fișier
    with open('example_data.json', 'w') as f:
        json.dump(data_structure, f, indent=2)
    
    print("✓ Fișier JSON de exemplu creat: example_data.json")

# Exemplu 3: Cum să antrenezi modelul
def example_training():
    """Exemplu de antrenare a modelului."""
    from train import train_model
    
    # Antrenează modelul cu parametrii default
    # Asigură-te că ai un fișier train.json valid
    try:
        model = train_model(
            json_path="train.json",  # Calea către datele tale
            epochs=5,                # Numărul de epoci
            batch_size=2,           # Dimensiunea batch-ului
            lr=1e-3                 # Learning rate
        )
        print("✓ Antrenarea completă!")
        
        # Salvează modelul antrenat
        import torch
        torch.save(model.state_dict(), 'prostate_segmentation_model.pth')
        print("✓ Model salvat în: prostate_segmentation_model.pth")
        
    except FileNotFoundError:
        print("⚠️ Fișierul train.json nu există!")
        print("Creează mai întâi un fișier JSON cu datele tale.")

# Exemplu 4: Cum să încarci și folosești un model antrenat
def example_inference():
    """Exemplu de inferență cu un model antrenat."""
    from U_net import UNet
    import torch
    from PIL import Image
    import torchvision.transforms as T
    
    # Creează modelul
    model = UNet(n_channels=3, n_classes=3)
    
    # Încarcă parametrii antrenați (dacă există)
    try:
        model.load_state_dict(torch.load('prostate_segmentation_model.pth'))
        print("✓ Model antrenat încărcat")
    except FileNotFoundError:
        print("⚠️ Model antrenat nu găsit. Folosesc model cu parametri aleatori.")
    
    model.eval()
    
    # Simulează o imagine de test
    dummy_image = torch.randn(1, 3, 256, 256)
    
    # Inferență
    with torch.no_grad():
        output = model(dummy_image)
        prediction = torch.argmax(output, dim=1)
    
    # Analizează rezultatul
    unique_classes = torch.unique(prediction)
    print(f"Clase detectate în imagine: {unique_classes.tolist()}")
    
    # Maparea claselor
    class_names = {0: "Fundal", 1: "Prostata", 2: "Zona periferică"}
    for cls in unique_classes:
        pixels = (prediction == cls).sum().item()
        total_pixels = prediction.numel()
        percentage = (pixels / total_pixels) * 100
        print(f"  {class_names[cls.item()]}: {pixels} pixeli ({percentage:.1f}%)")

if __name__ == "__main__":
    print("📖 Exemple de utilizare - Sistemul de Segmentare a Prostatei")
    print("=" * 65)
    
    print("\n1. 🧠 Exemplu de utilizare a modelului:")
    try:
        example_model_usage()
    except ImportError:
        print("⚠️ PyTorch nu este instalat. Rulează: pip install -r requirements.txt")
    
    print("\n2. 📊 Exemplu de pregătire a datelor:")
    example_data_preparation()
    
    print("\n3. 🏋️ Exemplu de antrenare:")
    print("Pentru a antrena modelul, asigură-te că ai:")
    print("  - Un fișier train.json cu datele tale")
    print("  - PyTorch instalat (pip install -r requirements.txt)")
    
    print("\n4. 🔍 Exemplu de inferență:")
    try:
        example_inference()
    except ImportError:
        print("⚠️ PyTorch nu este instalat pentru inferență.")
    
    print("\n💡 Pentru demonstrații interactive, rulează:")
    print("   python demo.py")