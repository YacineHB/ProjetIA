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
        self.binary_image = None
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
        # Binairisation de l'image
        channels = cv2.split(self.image)
        binary_images = []

        # Seuillage par canal (éviter la perte d'informations de couleur)
        for channel in channels:
            _, binary_channel = cv2.threshold(channel, 127, 255, cv2.THRESH_BINARY_INV)
            binary_images.append(binary_channel)

        # Fusionner les canaux binarisés (ou logique)
        binary_image = cv2.bitwise_or(binary_images[0], binary_images[1])
        binary_image = cv2.bitwise_or(binary_image, binary_images[2])
        return binary_image

    def detect_and_classify_objects(self):
        """Détecter et classer les objets dans l'image en couleur sans conversion en niveaux de gris."""
        if self.model is None:
            print("Aucun modèle de classification fourni.")
            return None, None

        self.binary_image = self.preprocess_image()

        # Trouver les contours dans l'image binarisée
        contours, _ = cv2.findContours(self.binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Dictionnaire pour stocker les comptages des classes détectées
        class_counts = defaultdict(int)
        detected_objects = []

        for contour in contours:
            if cv2.contourArea(contour) < 50:
                continue

            x, y, w, h = enlarge_contour(cv2.boundingRect(contour), top=15, left=2, right=2)
            letter_image = self.image[y:y + h, x:x + w]

            # Prédiction avec le modèle
            predicted_class = self.model.predict(letter_image)

            if predicted_class is not None:
                # Incrémenter le comptage de la classe prédite
                class_counts[predicted_class] += 1

                # Ajouter les coordonnées et la classe prédite
                detected_objects.append((x, y, w, h, predicted_class))

        return dict(sorted(class_counts.items())), detected_objects

    def display_results(self, class_counts, detected_objects):
        """Afficher les résultats de la détection et classification avec la même résolution que l'image d'origine."""
        try:
            self.display_binary_image()
            self.display_image_with_classes(detected_objects)
            self.display_image_with_annotations(detected_objects)
            self.display_classes_count(class_counts)
        except Exception as e:
            print(f"Une erreur s'est produite lors de l'affichage des résultats : {e}")

    def display_binary_image(self):
        """Afficher l'image binaire."""
        plt.figure(figsize=(self.binary_image.shape[1] / 100, self.binary_image.shape[0] / 100))
        plt.imshow(self.binary_image, cmap='gray')
        plt.axis('off')
        plt.show()

    def display_image_with_classes(self, detected_objects):
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

    def display_image_with_annotations(self, detected_objects):
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

    def display_classes_count(self, class_counts):
        """Afficher le nombre d'objets détectés par classe."""
        # afficher les classes prédites et le nombre de fois qu'elles apparaissent
        plt.figure(figsize=(10, 5))
        plt.bar(class_counts.keys(), class_counts.values())
        plt.xlabel("Classes")
        plt.ylabel("Nombre de lettres")
        plt.title("Classes détectées et leur nombre")
        plt.show()




