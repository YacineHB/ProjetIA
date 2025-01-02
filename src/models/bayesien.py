import os
import cv2
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


class BayesianClassifier:
    def __init__(self):
        self.feature_means = {}
        self.feature_variances = {}
        self.class_priors = {}
        self.classes = []

    def extract_features(self, image):
        """Extraire les caractéristiques d'une image : détection des contours et normalisation."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Seuillage pour binariser l'image (légèrement inversé pour améliorer la détection)
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)

        # Trouver les contours dans l'image binarisée
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extraire la caractéristique de chaque contour : utilisation de la zone de chaque contour
        features = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            letter_image = gray_image[y:y + h, x:x + w]
            resized_letter = cv2.resize(letter_image, (28, 28))  # Redimensionner à une taille fixe
            features.append(resized_letter.flatten())  # Aplatir l'image pour en faire un vecteur
        features = np.array(features)

        # Normalisation des caractéristiques pour chaque contour (lettre)
        if len(features) > 0:
            features = features / np.linalg.norm(features, axis=1, keepdims=True)

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
        features = self.extract_features(image)
        posteriors = {}

        for class_name in self.classes:
            # Calcul de la vraisemblance (log-vraisemblance)
            mean = self.feature_means[class_name]
            variance = self.feature_variances[class_name]
            likelihood = -0.5 * np.sum(((features - mean) ** 2) / variance + np.log(2 * np.pi * variance))
            # Ajouter la probabilité a priori (log-prior)
            posterior = likelihood + np.log(self.class_priors[class_name])
            posteriors[class_name] = posterior

        # Retourner la classe ayant la plus haute probabilité a posteriori
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

    def test(self, test_path):
        """Tester le classifieur et calculer la précision."""
        correct = 0
        total = 0

        # Parcourir les sous-dossiers du répertoire de test
        for class_name in os.listdir(test_path):
            class_folder_path = os.path.join(test_path, class_name)
            if os.path.isdir(class_folder_path):  # Vérifier si c'est un sous-dossier
                for img_name in os.listdir(class_folder_path):
                    img_path = os.path.join(class_folder_path, img_name)
                    if os.path.isfile(img_path):
                        image = cv2.imread(img_path)
                        if image is not None:
                            predicted_class = self.predict(image)
                            total += 1
                            if predicted_class == class_name:
                                correct += 1

        accuracy = correct / total if total > 0 else 0
        return accuracy


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