import cv2
import torch
import os
import numpy as np
from torchvision import transforms
from matplotlib import pyplot as plt
from src.models.bayesien import BayesianClassifier
from collections import defaultdict

class ObjectDetectionPipeline:
    def __init__(self, image_path, model=None):
        self.image_path = image_path
        self.image = None
        self.model = model  # Le modèle personnalisé à utiliser

    def load_model(self, model_path):
        """Charger le modèle pré-entrainé ou le classifieur bayésien."""
        if os.path.exists(model_path):
            # Charger un modèle bayésien si le chemin existe
            self.model = BayesianClassifier()  # Créer une instance du classifieur bayésien
            self.model.load_model(model_path)  # Charger les paramètres du modèle bayésien
            print(f"Modèle bayésien chargé depuis {model_path}")
        else:
            print(f"Aucun modèle trouvé à {model_path}. Un nouveau modèle sera créé.")

    def load_image(self):
        """Charge l'image à traiter."""
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise FileNotFoundError(f"L'image {self.image_path} est introuvable.")
        return self.image

    def preprocess_image(self):
        """Prétraiter l'image pour l'inférence."""
        # Convertir l'image en niveaux de gris pour la détection de caractéristiques par le classifieur bayésien
        image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return image_gray

    def detect_and_classify_objects(self):
        """Détecter et classer les objets dans l'image à l'aide du modèle."""
        if self.model is None:
            print("Aucun modèle de classification fourni.")
            return {}

        # Prétraiter l'image pour l'inférence (conversion en niveaux de gris)
        processed_image = self.preprocess_image()

        # Seuillage pour binariser l'image (légèrement inversé pour améliorer la détection)
        _, binary_image = cv2.threshold(processed_image, 127, 255, cv2.THRESH_BINARY_INV)

        # Trouver les contours dans l'image binarisée
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Dictionnaire pour stocker les comptages des classes détectées
        class_counts = defaultdict(int)
        detected_objects = []

        for contour in contours:
            # Ignorer les petits contours (bruit)
            if cv2.contourArea(contour) < 50:  # Limiter à des zones suffisamment grandes
                continue

            x, y, w, h = cv2.boundingRect(contour)
            letter_image = processed_image[y:y + h, x:x + w]
            resized_letter = cv2.resize(letter_image, (28, 28))  # Redimensionner à une taille fixe

            # Prédiction avec le modèle bayésien
            predicted_class = self.model.predict(resized_letter)

            # Incrémenter le comptage de la classe prédite
            class_counts[predicted_class] += 1

            # Ajouter les coordonnées et la classe prédite pour afficher un rectangle plus tard
            detected_objects.append((x, y, w, h, predicted_class))

        return dict(sorted(class_counts.items())), detected_objects

    def display_results(self, class_counts, detected_objects):
        """Afficher l'image avec la classe prédite et les rectangles autour des objets détectés."""
        image_copy = self.image.copy()

        # Affichage des rectangles autour des lettres et des classes
        for (x, y, w, h, predicted_class) in detected_objects:
            # Dessiner le rectangle autour de l'objet
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Afficher la classe prédite sur l'image
            cv2.putText(image_copy, f"{predicted_class}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Afficher l'image avec les résultats
        fig = plt.figure(figsize=(image_copy.shape[1] / 100, image_copy.shape[0] / 100))
        plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
        plt.title(f"Comptage des objets: {class_counts}")
        plt.axis('off')
        plt.show()
