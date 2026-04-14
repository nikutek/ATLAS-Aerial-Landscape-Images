# ATLAS — Aerial Landscape Image Classification

Projekt klasyfikacji zdjęć lotniczych/satelitarnych na 15 kategorii krajobrazów. Zrealizowany w trzech podejściach: własna sieć CNN (PyTorch), fine-tuning ResNet18 oraz model YOLOv11n-cls.

**Autorzy:** Mykhailo Zemliakov, Sebastian Cybul, Tomasz Okniński, Nikodem Goławski, Daniel Kadej

---

## Klasy

Model rozpoznaje 15 kategorii krajobrazów lotniczych:

| | | |
|---|---|---|
| Agriculture | Airport | Beach |
| City | Desert | Forest |
| Grassland | Highway | Lake |
| Mountain | Parking | Port |
| Railway | Residential | River |

Dataset pochodzi ze zbioru [ATLAS Aerial Landscape Images (Kaggle)](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset).

---

## Struktura repozytorium

```
ATLAS-Aerial-Landscape-Images/
├── simple_cnn/                  # Własna sieć CNN (PyTorch)
│   ├── checkpoints/             # Zapisane wagi modelu (.pth)
│   ├── results/                 # Wykresy krzywych uczenia i historia treningu
│   └── simple_cnn/
│       ├── model.py             # Architektura SimpleCNN
│       ├── train.py             # Skrypt trenujący
│       ├── dataset.py           # Ładowanie i augmentacja danych
│       ├── split_data.py        # Podział danych train/val/test
│       ├── compute_stats.py     # Obliczanie mean/std datasetu
│       ├── config.yaml          # Konfiguracja hiperparametrów
│       └── requirements.txt     # Zależności
└── yolo-v8/                     # Model YOLOv11 (i ResNet18)
    ├── runs/                    # Wyniki treningu YOLO (wagi, wykresy, macierz pomyłek)
    ├── main.py                  # Trening YOLOv11
    ├── predict.py               # Predykcja na obrazach
    ├── predictFromScreen.py     # Demo w czasie rzeczywistym z ekranu
    ├── distributeData.py        # Podział danych 70/15/15
    ├── yolo11n-cls.pt           # Wagi bazowe YOLOv11n-cls
    └── requirements.txt         # Zależności
```

---

## Modele i wyniki

### 1. SimpleCNN (PyTorch)

Własna konwolucyjna sieć neuronowa budowana iteracyjnie w 4 wersjach.

**Architektura końcowa (v4.0):**
- 4 bloki sekwencyjne: `Conv2d → BatchNorm2d → ReLU → MaxPool2d`
- Kanały: 16 → 32 → 64 → 128
- Warstwy klasyfikatora: `Linear(25088, 512) → ReLU → Dropout → Linear(512, 15)`
- Łącznie ~12,9M parametrów

**Augmentacja danych:**
```
RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(30°),
ColorJitter(brightness=0.1, contrast=0.1), Normalize(mean, std)
```

**Ewolucja modelu:**

| Wersja | Zmiany | Val Accuracy |
|--------|--------|-------------|
| v1.0 | 3 bloki Sequential, wyjście 100MB | ~82% |
| v2.0 | Zmiana argumentów MaxPool, Dropout, ReduceLROnPlateau | ~85% |
| v3.0 | Dodanie 4. bloku Sequential | ~90% |
| v4.0 | Optymalizacja rozmiaru wyjścia (50MB) | ~90% |

**Konfiguracja treningu** (`config.yaml`):
- `img_size: 224`, `batch_size: 64`, `epochs: 50`
- `learning_rate: 0.001`, scheduler co 20 epok (gamma 0.2)

### 2. ResNet18 (transfer learning)

Fine-tuning pretrenowanego ResNet18 z podmienioną warstwą `fc`.

| Wersja | Zmiany | Val Accuracy |
|--------|--------|-------------|
| v1.0 | Stały wysoki LR | ~95% (niestabilny trening) |
| v2.0 | Scheduler LR | 93.4% |
| v3.0 | 30 epok treningu | **97%** |

### 3. YOLOv11n-cls

Transfer learning na bazie YOLOv11n-cls (nano wariant Ultralytics).

**Augmentacja:** HSV (h=0.015, s=0.7, v=0.4), obroty ±15°, skalowanie 50%, odbicia poziome i pionowe (p=0.5)

**Wyniki po 15 epokach:**
- Top-1 Accuracy: **~97%**
- Top-5 Accuracy: **~99.99%**
- Early stopping: patience=10 epok

---

## Szybki start

### SimpleCNN

```bash
cd simple_cnn/simple_cnn

# Instalacja zależności
pip install -r requirements.txt

# 1. Pobierz dataset z Kaggle i umieść w simple_cnn/data/raw/
#    (foldery z nazwami klas: Agriculture/, Airport/, ...)

# 2. Oblicz statystyki datasetu (opcjonalnie)
python compute_stats.py

# 3. Podziel dane na train/val/test
python split_data.py

# 4. Trenuj model
python train.py
```

### YOLOv11

```bash
cd yolo-v8

# Instalacja zależności
pip install -r requirements.txt

# 1. Pobierz dataset z Kaggle i umieść w yolo-v8/data/
#    (foldery z nazwami klas)

# 2. Podziel dane (70% train / 15% val / 15% test)
python distributeData.py

# 3. Trenuj model
python main.py

# 4. Predykcja na obrazie
python predict.py

# 5. Demo w czasie rzeczywistym z ekranu (Google Maps satelita)
python predictFromScreen.py
```

### Demo na żywo (`predictFromScreen.py`)

Aplikacja przechwytuje fragment ekranu (512×512 px z lewej połowy monitora) co 0,5 sekundy i wyświetla nazwę klasy wraz z pewnością predykcji. Idealna do demonstracji na Google Maps w trybie satelitarnym.

---

## Pobieranie danych

1. Wejdź na [Kaggle — SkyView Aerial Landscape Dataset](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset)
2. Pobierz i rozpakuj archiwum
3. Umieść foldery klas w `simple_cnn/data/raw/` (SimpleCNN) lub `yolo-v8/data/` (YOLO)

Oczekiwana struktura:
```
data/
├── Agriculture/
├── Airport/
├── Beach/
├── City/
├── Desert/
├── Forest/
├── Grassland/
├── Highway/
├── Lake/
├── Mountain/
├── Parking/
├── Port/
├── Railway/
├── Residential/
└── River/
```

---

## Porównanie modeli

| Model | Parametry | Epoki | Val Accuracy |
|-------|-----------|-------|-------------|
| SimpleCNN v4.0 | ~12.9M | 60 | ~90% |
| ResNet18 v3.0 | ~11.2M | 30 | ~97% |
| YOLOv11n-cls | ~2.6M | 15 | ~97% |

YOLOv11 osiąga wyniki porównywalne z ResNet18 przy znacznie krótszym czasie treningu i mniejszej liczbie parametrów, dzięki zastosowaniu transfer learningu na pretrenowanych wagach Ultralytics.
