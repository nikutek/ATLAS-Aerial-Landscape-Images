# ATLAS - Aerial Landscape Image Classification

Projekt klasyfikacji obrazów lotniczych krajobrazu przy użyciu YOLOv8.

## 📋 Klasy
Model rozpoznaje 8 typów krajobrazów:
- Agriculture 
- Beach
- City
- Desert
- Forest
- Mountain
- Railway
- Residential

## 🚀 Instalacja i uruchomienie

### 1. Sklonuj repozytorium
```bash
git clone https://github.com/nikutek/ATLAS-Aerial-Landscape-Images/edit/main/README.md)
cd ATLAS-Aerial-Landscape-Images/yolo-v8
```
### 2. Stwórz środowisko wirtualne
```bash
python -m venv .venv
```
### 3. Aktywuj środowisko
**Windows:**
```bash
.venv\Scripts\activate
```
**Linux/Mac:**
```bash
source .venv/bin/activate
```
### 4. Zainstaluj zależności
```bash
pip install -r requirements.txt
```
### 5. Pobierz dane z Kaggle
Pobierz dataset z Kaggle:
- Link: (https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset?resource=download)
- Rozpakuj pliki do folderu `data/`

Struktura powinna wyglądać tak:
```
yolo-v8/
├── data/
│   ├── Agriculture/
│   │   ├── obraz1.jpg
│   │   ├── obraz2.jpg
│   │   └── ...
│   ├── Beach/
│   ├── City/
│   ├── Desert/
│   ├── Forest/
│   ├── Mountain/
│   ├── Railway/
│   └── Residential/
```

### 6. Przygotuj dane treningowe
```bash
python distributeData.py
```

Ten skrypt automatycznie podzieli dane na:
- 70% train
- 15% validation
- 15% test

### 7. Trenowanie modelu
Aktualny model jest już wytrenowany. Wagi znajdują się w `runs/classify*/weights/best.pt`

## 📁 Struktura projektu
```
yolo-v8/
├── data/                  # Dane źródłowe (nie w repo)
├── train/                 # Dane treningowe (nie w repo)
├── val/                   # Dane walidacyjne (nie w repo)
├── test/                  # Dane testowe (nie w repo)
├── runs/                  # Wyniki treningów
│   └── classify*/weights/
│       ├── best.pt       # Najlepszy model (w repo)
│       └── last.pt       # Ostatni checkpoint (w repo)
├── distributeData.py      # Skrypt podziału danych
├── requirements.txt       # Zależności Python
└── .gitignore
```

## 📊 Wyniki

<img width="1200" height="1200" alt="image" src="https://github.com/user-attachments/assets/f8d3d033-e713-45ee-9d14-fda7aac6a5b4" />
<img width="3000" height="2250" alt="image" src="https://github.com/user-attachments/assets/69d90b53-fe59-4530-8d4b-171a664c267d" />


## 📝 Autorzy
Nikodem Goławski
Tomasz Okniński
Daniel Kadej
Sebastian Cybul
Mykhailo Z
