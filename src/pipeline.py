import cv2
import os
from matplotlib import pyplot as plt
from src.classifiers.abstractclassifier import Classifier
from src.utils import enlarge_contour
from collections import defaultdict

class ObjectDetectionPipeline:
    def __init__(self, image_path, model=None):
        self.image_path = image_path
        self.image = None
        self.model = model  # Le modèle personnalisé à utiliser

    def load_model(self, model_path : str, instance_classifier : Classifier=None):
        """Charger le modèle pré-entrainé ou le classifieur bayésien."""
        if os.path.exists(model_path):
            # Charger un modèle bayésien si le chemin existe
            self.model = instance_classifier  # Créer une instance du classifieur
            print(f"Chargement du modèle bayésien depuis {model_path}")
            self.model.load_model(model_path)  # Charger les paramètres du modèle
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

            x, y, w, h = enlarge_contour(cv2.boundingRect(contour), top=15, left=2, right=2)
            letter_image = processed_image[y:y + h, x:x + w]

            # Prédiction avec le modèle bayésien
            predicted_class = self.model.predict(letter_image)

            # Incrémenter le comptage de la classe prédite
            class_counts[predicted_class] += 1

            # Ajouter les coordonnées et la classe prédite pour afficher un rectangle plus tard
            detected_objects.append((x, y, w, h, predicted_class))

        return dict(sorted(class_counts.items())), detected_objects

    def display_results(self, class_counts, detected_objects):
        """Afficher les résultats de la détection et classification avec la même résolution que l'image d'origine."""
        self.display_image_with_classes(class_counts, detected_objects)
        self.display_image_with_annotations(class_counts, detected_objects)

    def display_image_with_classes(self, class_counts, detected_objects):
        """Afficher l'image avec les classes prédites."""
        image_with_classes_only = self.image.copy()  # Copie de l'image d'origine

        # Boucle sur les objets détectés pour afficher les rectangles et les textes
        for (x, y, w, h, predicted_class) in detected_objects:
            # Vérifier si la classe prédite a "_" à la fin
            if predicted_class[-1] == "_":
                text = predicted_class.split("_")[0].upper()
            else:
                text = predicted_class.lower()

            # --- Pour l'image avec seulement les classes ---
            # Effacer l'ancienne lettre en remplissant la région avec du blanc (ou autre couleur de fond)
            cv2.rectangle(image_with_classes_only, (x, y), (x + w, y + h), (255, 255, 255), -1)

            # Calculer la position pour centrer le texte dans la région de la lettre
            font_scale = 0.7
            font_thickness = 2
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = y + (h + text_size[1]) // 2

            # Ajouter le texte de la classe prédite
            cv2.putText(image_with_classes_only, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

        # Affichage de l'image avec les classes à la même résolution que l'image originale
        fig = plt.figure(figsize=(image_with_classes_only.shape[1] / 100, image_with_classes_only.shape[0] / 100))
        plt.imshow(cv2.cvtColor(image_with_classes_only, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # Désactiver les axes
        plt.show()  # Afficher l'image

    def display_image_with_annotations(self, class_counts, detected_objects):
        """Afficher l'image avec les annotations (rectangles et textes)."""
        image_with_annotations = self.image.copy()  # Copie de l'image d'origine
        for (x, y, w, h, predicted_class) in detected_objects:
            # Vérifier si la classe prédite a "_" à la fin
            if predicted_class[-1] == "_":
                text = predicted_class.split("_")[0].upper()
            else:
                text = predicted_class.lower()

            # --- Pour l'image avec annotations ---
            # Dessiner le rectangle autour de la lettre détectée
            cv2.rectangle(image_with_annotations, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Ajouter le texte à côté de la lettre
            cv2.putText(image_with_annotations, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Affichage de l'image avec les annotations à la même résolution que l'image originale
        fig = plt.figure(figsize=(image_with_annotations.shape[1] / 100, image_with_annotations.shape[0] / 100))
        plt.imshow(cv2.cvtColor(image_with_annotations, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # Désactiver les axes
        plt.show()  # Afficher l'image




