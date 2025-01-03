#Abstract Classifier class

import abc

class Classifier:

    def __init__(self):
        pass

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