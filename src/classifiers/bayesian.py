import os
import cv2
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from src.classifiers.abstractclassifier import Classifier


class BayesianClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.feature_means = {}
        self.feature_variances = {}
        self.class_priors = {}
        self.classes = []
        self.hog = cv2.HOGDescriptor(
            _winSize=(28, 28),  # Taille de la fenêtre (même que l'image redimensionnée)
            _blockSize=(8, 8),  # Taille des blocs
            _blockStride=(4, 4),  # Pas entre les blocs
            _cellSize=(8, 8),  # Taille des cellules
            _nbins=9  # Nombre de bins pour l'histogramme des orientations
        )

    def extract_features(self, image):
        """Extraire les caractéristiques HOG d'une image."""
        # Vérifier si l'image est déjà en niveaux de gris
        if len(image.shape) == 3 and image.shape[2] == 3:  # Image couleur (3 canaux)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image  # Si l'image est déjà en niveaux de gris, on l'utilise telle quelle

        # Seuillage adaptatif pour mieux gérer les variations de luminosité
        binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY_INV, 11, 2)

        # Trouver les contours dans l'image binarisée
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        features = []
        for contour in contours:
            if cv2.contourArea(contour) < 22:  # Ignorer les petits contours (bruit)
                continue

            x, y, w, h = cv2.boundingRect(contour)
            letter_image = gray_image[y:y + h, x:x + w]
            letter_image = cv2.resize(letter_image, (28, 28))  # Redimensionner à une taille fixe

            # Extraire les caractéristiques HOG
            hog_features = self.hog.compute(letter_image)

            # Ajouter les caractéristiques au tableau global
            features.append(hog_features.flatten())

        features = np.array(features)

        # Normalisation des caractéristiques pour chaque contour (lettre)
        norms = np.linalg.norm(features, axis=1, keepdims=True)  # Calcul de la norme sur chaque ligne (chaque image)
        features = features / np.where(norms > 1e-6, norms, 1)  # Remplacer les normes nulles par 1

        return features

    def train(self, catalog_path):
        """Entraîner ou mettre à jour le classifieur Bayesien avec des images de lettres du catalogue."""
        class_features = defaultdict(list)
        total_images = 0

        # Parcourir les sous-dossiers du catalogue (chaque sous-dossier correspond à une classe)
        for class_name in os.listdir(catalog_path):
            class_folder_path = os.path.join(catalog_path, class_name)
            if os.path.isdir(class_folder_path):  # Vérifier si c'est un sous-dossier
                if class_name not in self.classes:
                    self.classes.append(class_name)
                # Parcourir les fichiers dans le sous-dossier de la classe
                for img_name in os.listdir(class_folder_path):
                    img_path = os.path.join(class_folder_path, img_name)
                    if os.path.isfile(img_path):
                        image = cv2.imread(img_path)
                        if image is not None:
                            features = self.extract_features(image)
                            for feature in features:
                                class_features[class_name].append(feature)
                            total_images += 1

        # Calcul des moyennes, variances et des priorités
        for class_name in self.classes:
            if class_name in class_features:  # Assurez-vous que des images existent pour la classe
                features = np.array(class_features[class_name])  # Convertir en array numpy
                self.feature_means[class_name] = np.mean(features, axis=0)
                self.feature_variances[class_name] = np.var(features, axis=0) + 1e-6  # Eviter la division par zéro
                self.class_priors[class_name] = len(features) / total_images

        print("Classes entraînées ou mises à jour :", self.classes)

    def predict(self, image):
        """Prédire la classe d'une image en fonction des caractéristiques extraites."""
        # Poids des rotations
        rotation_weights = {
            0: 1.0,  # Importance maximale pour l'angle 0
            90: 0.5,  # Réduction de poids pour les autres angles
            180: 0.5,
            270: 0.5
        }

        posteriors = {}

        for rotation, weight in rotation_weights.items():  # Parcourir les rotations et leurs poids
            k = rotation // 90  # Convertir degrés en nombre de rotations pour np.rot90
            rotated_image = np.rot90(image, k)
            features = self.extract_features(rotated_image)

            for class_name in self.classes:
                mean = self.feature_means[class_name]
                variance = self.feature_variances[class_name]
                prior = self.class_priors[class_name]

                # Calculer la vraisemblance logarithmique (log de la densité gaussienne)
                likelihood = -0.5 * np.sum(((features - mean) ** 2) / variance + np.log(2 * np.pi * variance))
                posterior = likelihood + np.log(prior)

                # Appliquer le poids de la rotation
                weighted_posterior = posterior * (1 - weight * 0.5)

                # Conserver la meilleure probabilité pondérée pour chaque classe
                if class_name not in posteriors:
                    posteriors[class_name] = weighted_posterior
                else:
                    posteriors[class_name] = max(posteriors[class_name], weighted_posterior)


        # Retourner la classe ayant la probabilité maximale
        return max(posteriors, key=posteriors.get)

    def save_model(self, model_path):
        """Sauvegarder les paramètres du modèle avec PyTorch."""
        model_data = {
            "feature_means": self.feature_means,
            "feature_variances": self.feature_variances,
            "class_priors": self.class_priors,
            "classes": self.classes
        }
        torch.save(model_data, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path):
        """Charger les paramètres du modèle avec PyTorch."""
        if os.path.exists(model_path):
            model_data = torch.load(model_path)
            self.feature_means = model_data["feature_means"]
            self.feature_variances = model_data["feature_variances"]
            self.class_priors = model_data["class_priors"]
            self.classes = model_data["classes"]
            print(f"Model loaded from {model_path}")
        else:
            print(f"Aucun modèle trouvé à {model_path}. Un nouveau modèle sera créé.")


    def visualize_model(self):
        """Visualiser les moyennes des caractéristiques pour chaque classe"""
        if not self.classes:
            print("Aucune classe disponible pour la visualisation.")
            return

        for class_name in self.classes:
            mean_features = self.feature_means[class_name]

            plt.figure(figsize=(10, 4))
            plt.title(f'Moyenne des caractéristiques pour la classe: {class_name}')
            plt.plot(mean_features)  # Utiliser numpy pour l'affichage
            plt.xlabel('Index des caractéristiques')
            plt.ylabel('Valeur moyenne')
            plt.grid(True)
            plt.show()