import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

class ObjectDetectionPipeline:
    def __init__(self, image_path, catalog_path):
        self.image_path = image_path
        self.catalog_path = catalog_path
        self.image = None
        self.catalog_images = {}

    def load_image(self):
        """Charge l'image à traiter."""
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise FileNotFoundError(f"L'image {self.image_path} est introuvable.")
        return self.image

    def load_catalog(self):
        """Charge les objets à rechercher dans le catalogue."""
        for file_name in os.listdir(self.catalog_path):
            if file_name.endswith('.png') or file_name.endswith('.jpg'):
                object_image = cv2.imread(os.path.join(self.catalog_path, file_name), cv2.IMREAD_GRAYSCALE)
                self.catalog_images[file_name] = object_image

    def preprocess_image(self):
        """Convertir l'image en niveaux de gris tout en préservant la qualité."""
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def rotate_image(self, image, angle):
        """Effectuer une rotation de l'image."""
        if angle == 0:
            return image
        elif angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def detect_objects(self, processed_image):
        """Cherche les objets dans l'image à l'aide de template matching."""
        found_locations = {}

        for obj_name, object_image in self.catalog_images.items():
            locations = []
            object_height, object_width = object_image.shape

            for angle in [0, 90, 180, 270]:
                rotated_object = self.rotate_image(object_image, angle)
                result = cv2.matchTemplate(processed_image, rotated_object, cv2.TM_CCOEFF_NORMED)
                threshold = 0.7
                loc = np.where(result >= threshold)

                for pt in zip(*loc[::-1]):
                    locations.append((pt[0], pt[1], object_width, object_height))

            found_locations[obj_name] = locations

        return found_locations

    def display_results(self, found_locations):
        """Afficher les objets détectés sur l'image sans altérer la qualité de l'image."""
        image_copy = self.image.copy()
        color_map = {obj_name: (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                     for obj_name in found_locations.keys()}

        for obj_name, locations in found_locations.items():
            for (x, y, w, h) in locations:
                color = color_map[obj_name]
                cv2.rectangle(image_copy, (x, y), (x + w, y + h), color, 2)

        fig = plt.figure(figsize=(image_copy.shape[1] / 100, image_copy.shape[0] / 100))
        plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
        plt.title("Objects Detected")
        plt.axis('off')
        plt.show()

    def count_objects(self, found_locations):
        """Compter les objets détectés pour chaque type."""
        counts = {}
        for obj_name, locations in found_locations.items():
            counts[obj_name] = len(locations)
        return counts
