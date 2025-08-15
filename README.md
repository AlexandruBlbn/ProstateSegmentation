# ProstateSegmentation

Implementarea rețelei U-net pentru segmentarea prostatei pe multiple clase folosind PyTorch.

## Cum funcționează?

Acest proiect implementează o rețea neurală U-Net pentru segmentarea automată a prostatei în imagini medicale. Sistemul clasifică fiecare pixel în una din cele 3 clase:

- **Clasa 0**: Fundal (Background)
- **Clasa 1**: Prostata (Prostate)  
- **Clasa 2**: Zona periferică (Peripheral Zone)

### Arhitectura U-Net

U-Net este o arhitectură de rețea neurală convoluțională specializată pentru segmentarea imaginilor, cu următoarele componente:

1. **Encoder (Downsampling)**: Reduce dimensiunea imaginii și extrage caracteristici
2. **Bottleneck**: Nivelul cel mai profund al rețelei
3. **Decoder (Upsampling)**: Reconstruiește imaginea segmentată la dimensiunea originală
4. **Skip connections**: Conectează nivelurile encoder-ului cu decoder-ul pentru păstrarea detaliilor

## Instalare și Configurare

### Cerințe de sistem
- Python 3.8+
- CUDA (opțional, pentru accelerare GPU)

### Instalare dependințe
```bash
pip install -r requirements.txt
```

### Dataset
Link dataset: https://www.kaggle.com/datasets/haithem1999/prostate-annotated-dataset-for-image-segmentation

Formatul așteptat pentru dataset este un fișier JSON cu următoarea structură:
```json
{
  "Pacient": [
    {
      "data": "path/to/image.png",
      "label": "path/to/mask.png"
    }
  ]
}
```

## Utilizare

### 1. Antrenarea modelului
```python
from train import train_model

# Antrenează modelul cu parametrii default
model = train_model("path/to/train.json", epochs=10, batch_size=4)
```

### 2. Încărcarea datelor
```python
from dataloader import get_dataloaders

# Obține dataloader-ele pentru antrenare și validare
train_loader, val_loader = get_dataloaders("train.json", batch_size=4, augment=True)
```

### 3. Utilizarea modelului U-Net
```python
from U_net import UNet
import torch

# Creează modelul
model = UNet(n_channels=3, n_classes=3)

# Pentru imagini RGB (3 canale) și 3 clase de segmentare
input_tensor = torch.randn(1, 3, 256, 256)  # Batch size=1, 3 channels, 256x256 pixels
output = model(input_tensor)  # Output: (1, 3, 256, 256) - probabilități pentru fiecare clasă
```

## Structura Proiectului

```
├── U_net.py            # Implementarea arhitecturii U-Net
├── dataloader.py       # Încărcarea și preprocesarea datelor
├── train.py            # Antrenarea și evaluarea modelului
├── demo.py             # Script de demonstrație interactiv
├── usage_examples.py   # Exemple simple de utilizare
├── requirements.txt    # Dependințele proiectului
└── README.md           # Documentația proiectului
```

## Exemple și Demo

### 🎮 Demo interactiv
Pentru o demonstrație completă a tuturor componentelor:
```bash
python demo.py
```

### 📖 Exemple de cod
Pentru exemple simple de utilizare a fiecărei componente:
```bash
python usage_examples.py
```

Acest script include exemple pentru:
- Crearea și utilizarea modelului U-Net
- Pregătirea datelor în formatul corect
- Antrenarea modelului
- Inferența cu un model antrenat

## Metrici de Evaluare

Modelul utilizează **Dice Score** pentru evaluarea performanței:
- Dice Score pe clasă: Măsoară overlap-ul dintre predicție și ground truth pentru fiecare clasă
- Mean Dice Score: Media tuturor claselor

## Parametri de Antrenare

- **Learning Rate**: 1e-3 (Adam optimizer)
- **Batch Size**: 4 (configurabil)
- **Epochs**: 10 (configurabil)
- **Loss Function**: CrossEntropyLoss
- **Data Augmentation**: RandomHorizontalFlip, RandomRotation (30°)

## Framework
- **PyTorch**: 2.0+
- **Python**: 3.8+
