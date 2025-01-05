import os
import cv2
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from src.classifiers.abstractclassifier import Classifier


class KMeansClassifier(Classifier):
    def __init__(self, num_clusters=62):
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

        # Normalisation pour éviter les problèmes de gradient
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / np.where(norms > 1e-6, norms, 1)

        return features

    def initialize_centroids(self, features):
        """Initialiser les centroïdes avec K-means++."""
        # Convertir les caractéristiques en un tableau numpy pour une manipulation plus facile
        features_np = features.numpy()
        centroids = [features_np[np.random.choice(features_np.shape[0])]]  # Choix aléatoire initial

        for _ in range(1, self.num_clusters):
            distances = np.min([np.linalg.norm(features_np - c, axis=1) for c in centroids], axis=0)
            probabilities = distances / distances.sum()
            new_centroid = features_np[np.random.choice(features_np.shape[0], p=probabilities)]
            centroids.append(new_centroid)

        # Convertir les centroïdes en tensor PyTorch
        return torch.tensor(np.array(centroids), dtype=torch.float32)

    def assign_clusters(self, features, centroids):
        """Assigner chaque point au centroïde le plus proche."""
        distances = torch.cdist(features, centroids)
        cluster_assignments = torch.argmin(distances, dim=1)
        return cluster_assignments

    def update_centroids(self, features, cluster_assignments):
        """Mettre à jour les centroïdes en calculant la moyenne des points assignés à chaque cluster."""
        centroids = []
        for i in range(self.num_clusters):
            cluster_points = features[cluster_assignments == i]
            if len(cluster_points) == 0:
                # Réinitialiser le centroïde si le cluster est vide
                centroids.append(features[torch.randint(0, features.size(0), (1,))])
            else:
                centroids.append(cluster_points.mean(dim=0))
        return torch.stack(centroids)

    def train(self, catalog_path, num_iterations=100, tol=1e-4):
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
        for iteration in range(num_iterations):
            cluster_assignments = self.assign_clusters(all_features, self.centroids)
            new_centroids = self.update_centroids(all_features, cluster_assignments)

            # Vérifier la convergence
            centroid_shift = torch.norm(self.centroids - new_centroids, dim=1).max()
            self.centroids = new_centroids
            if centroid_shift < tol:
                print(f"Convergence atteinte après {iteration + 1} itérations.")
                break

        # Sauvegarder les assignations de clusters pour la visualisation
        self.labels_ = cluster_assignments  # Enregistrer les étiquettes
        self.data = all_features  # Enregistrer les données pour la visualisation

        print("Centroids calculated and clusters assigned.")


    def predict(self, image):
        """Prédire la classe d'une image basée sur l'appartenance à un cluster."""
        features = self.extract_features(image)  # Extraire les caractéristiques
        features = torch.tensor(features, dtype=torch.float32)

        # Vérifier que les caractéristiques ne sont pas vides
        if features.size(0) == 0:
            raise ValueError("Aucune caractéristique extraite de l'image. Vérifiez les images d'entrée.")

        # Assigner les clusters à la nouvelle image
        cluster_assignments = self.assign_clusters(features, self.centroids)

        # Si plusieurs points sont assignés, prendre le premier (ou définir une logique différente)
        if cluster_assignments.dim() > 0 and cluster_assignments.size(0) > 1:
            print("Plusieurs clusters assignés, en prenant le premier.")
            cluster_assignment = cluster_assignments[0].item()  # Prenez le premier cluster
        else:
            cluster_assignment = cluster_assignments.item()

        # Retourner la classe associée au cluster
        return self.classes[cluster_assignment]

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
        """
        Visualise les clusters et les centroïdes dans un plan 2D.
        """
        if self.centroids is None or not hasattr(self, "labels_"):
            raise AttributeError("Le modèle doit être entraîné avant toute visualisation.")

        # Projection des données sur les deux premières dimensions
        X_2d = self.data[:, :2].numpy()  # Utilise les deux premières dimensions
        centroids_2d = self.centroids[:, :2].numpy()  # Centroides dans les deux premières dimensions
        labels = self.labels_.numpy()  # Convertir les étiquettes en tableau numpy

        # Visualisation des points
        plt.figure(figsize=(10, 8))
        for cluster in range(self.num_clusters):
            cluster_points = X_2d[labels == cluster]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}", s=10)

        # Visualisation des centroïdes
        plt.scatter(
            centroids_2d[:, 0], centroids_2d[:, 1],
            color='black', marker='x', s=100, label='Centroids'
        )

        plt.title("Visualisation des clusters en 2D")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend(loc='best', fontsize='small')
        plt.grid(True)
        plt.show()

