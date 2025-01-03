import os
import cv2
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from src.classifiers.abstractclassifier import Classifier


class KMeansClassifier(Classifier):
    def __init__(self, num_clusters=3):
        super().__init__()
        self.num_clusters = num_clusters
        self.centroids = None
        self.classes = []
        self.hog = cv2.HOGDescriptor(
            _winSize=(28, 28),
            _blockSize=(8, 8),
            _blockStride=(4, 4),
            _cellSize=(8, 8),
            _nbins=9
        )

    def extract_features(self, image):
        """Extraire les caractéristiques HOG d'une image."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY_INV, 11, 2)

        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        features = []
        for contour in contours:
            if cv2.contourArea(contour) < 22:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            letter_image = gray_image[y:y + h, x:x + w]
            letter_image = cv2.resize(letter_image, (28, 28))

            hog_features = self.hog.compute(letter_image)
            features.append(hog_features.flatten())

        features = np.array(features)

        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / np.where(norms > 1e-6, norms, 1)

        return features

    def initialize_centroids(self, features):
        """Initialiser les centroïdes aléatoirement à partir des points de données."""
        indices = torch.randperm(features.size(0))[:self.num_clusters]
        centroids = features[indices]
        return centroids

    def assign_clusters(self, features, centroids):
        """Assigner chaque point au centroid le plus proche."""
        distances = torch.cdist(features, centroids)
        cluster_assignments = torch.argmin(distances, dim=1)
        return cluster_assignments

    def update_centroids(self, features, cluster_assignments):
        """Mettre à jour les centroïdes en calculant la moyenne des points assignés à chaque cluster."""
        centroids = torch.stack([features[cluster_assignments == i].mean(dim=0) for i in range(self.num_clusters)])
        return centroids

    def train(self, catalog_path, num_iterations=100):
        """Entraîner le modèle KMeans avec des images de lettres du catalogue."""
        class_features = defaultdict(list)
        total_images = 0

        for class_name in os.listdir(catalog_path):
            class_folder_path = os.path.join(catalog_path, class_name)
            if os.path.isdir(class_folder_path):
                if class_name not in self.classes:
                    self.classes.append(class_name)
                for img_name in os.listdir(class_folder_path):
                    img_path = os.path.join(class_folder_path, img_name)
                    if os.path.isfile(img_path):
                        image = cv2.imread(img_path)
                        if image is not None:
                            features = self.extract_features(image)
                            for feature in features:
                                class_features[class_name].append(feature)
                            total_images += 1

        # Convertir les caractéristiques en un tensor PyTorch
        all_features = []
        for class_name in self.classes:
            features = np.array(class_features[class_name])
            all_features.append(features)

        all_features = torch.tensor(np.vstack(all_features), dtype=torch.float32)

        # Initialiser les centroïdes
        self.centroids = self.initialize_centroids(all_features)

        # Effectuer l'algorithme KMeans
        for _ in range(num_iterations):
            cluster_assignments = self.assign_clusters(all_features, self.centroids)
            self.centroids = self.update_centroids(all_features, cluster_assignments)

        print("Centroids calculated and clusters assigned.")

    def predict(self, image):
        """Prédire la classe d'une image basée sur l'appartenance à un cluster."""
        features = self.extract_features(image)
        features = torch.tensor(features, dtype=torch.float32)

        # Assigner les clusters à la nouvelle image
        cluster_assignments = self.assign_clusters(features, self.centroids)

        # Vérifier si cluster_assignments est un tensor avec plus d'un élément
        if cluster_assignments.dim() > 0:
            cluster_assignments = cluster_assignments[0]  # Prenez le premier élément si plus d'un cluster est assigné

        # Retourner la classe associée au cluster
        return self.classes[cluster_assignments.item()]

    def save_model(self, model_path):
        """Sauvegarder les paramètres du modèle avec PyTorch."""
        model_data = {
            "centroids": self.centroids,
            "classes": self.classes
        }
        torch.save(model_data, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path):
        """Charger les paramètres du modèle avec PyTorch."""
        if os.path.exists(model_path):
            model_data = torch.load(model_path)
            self.centroids = model_data["centroids"]
            self.classes = model_data["classes"]
            print(f"Model loaded from {model_path}")
        else:
            print(f"Aucun modèle trouvé à {model_path}. Un nouveau modèle sera créé.")

    def visualize_model(self):
        """Visualiser les centres des clusters."""
        if self.centroids is None:
            print("Centroids not available for visualization.")
            return

        plt.figure(figsize=(10, 4))
        plt.title(f'Centroids of Clusters')
        for i, centroid in enumerate(self.centroids):
            plt.plot(centroid.numpy(), label=f'Cluster {i}')
        plt.xlabel('Index des caractéristiques')
        plt.ylabel('Valeur moyenne')
        plt.legend()
        plt.grid(True)
        plt.show()
