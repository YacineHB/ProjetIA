#Abstract Classifier class

import abc
import cv2

class Classifier(abc.ABC):

    def __init__(self):
        self.hog = cv2.HOGDescriptor(
            _winSize=(28, 28),  # Taille de la fenêtre (même que l'image redimensionnée)
            _blockSize=(8, 8),  # Taille des blocs
            _blockStride=(4, 4),  # Pas entre les blocs
            _cellSize=(8, 8),  # Taille des cellules
            _nbins=9  # Nombre de bins pour l'histogramme des orientations
        )

    @abc.abstractmethod
    def train(self, catalog_path):
        pass

    @abc.abstractmethod
    def predict(self, image):
        pass

    @abc.abstractmethod
    def save_model(self, model_path):
        pass

    @abc.abstractmethod
    def load_model(self, model_path):
        pass

    @abc.abstractmethod
    def visualize_model(self):
        pass