# ProjetIA - Détécteur de texte et de formes

## Installation des dépendances

Ce projet nécessite Python 3.8 ou une version ultérieure. Pour installer les dépendances nécessaires, exécutez la commande suivante dans votre terminal :

```bash
pip install torch torchvision torchaudio opencv-python numpy matplotlib
```

---

## Utilisation du detecteur de texte/formes (main.py)

### Pour exécuter le programme, exécutez la commande suivante dans votre terminal :

```bash
python main.py <chemin_vers_image> <chemin_vers_modele> <type de modele(bayesian/kmeans)>
```

exemple :

```bash
python main.py "data/plans/page.png" "models/bayesian_model.pth" "bayesian"
```

### ou lancer le fichier `main.py` dans votre IDE.

Vous pouvez également modifier les paramètres de détection dans le fichier `main.py`:
    
```python
# Paramètres de détection
image_path = "data/plans/page.png" # Chemin vers l'image à analyser
model_path = "models/bayesian_model.pth" # Chemin vers le modèle à utiliser
model_type = "bayesian" # Type de modèle à utiliser (bayesian/kmeans)
```

---

## Utilisation de l'entraîneur de modèle (train.py)

### Pour entraîner un modèle, exécutez la commande suivante dans votre terminal :

```bash
python train.py <chemin_vers_catalogue> <chemin_sauvegarde_modele> <type de modele(bayesian/kmeans)>
```

exemple :

```bash
python train.py "data/catalogue" "models/bayesian_model.pth" "bayesian"
```

### ou lancer le fichier `train.py` dans votre IDE.

Vous pouvez également modifier les paramètres d'entraînement dans le fichier `train.py`:
    
```python
# Paramètres d'entraînement
catalogue_path = "data/catalogue" # Chemin vers le catalogue d'images
model_path = "models/bayesian_model.pth" # Chemin de sauvegarde du modèle
model_type = "bayesian" # Type de modèle à entraîner (bayesian/kmeans)
```

---

## Auteurs

- [DIEUMEGARD Bilal]() : Développeur
- [HBADA Yacine]() : Développeur
