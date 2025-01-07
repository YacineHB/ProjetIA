from src.classifiers.bayesian import BayesianClassifier
from src.classifiers.kmeans import KMeansClassifier
from src.pipeline import ObjectDetectionPipeline
from src.utils import evaluate_performance
import time
import sys

image_path = "data/plans/page.png" # Choisir le chemin de l'image
model_path = "models/bayesian_model.pth" # Choisir le chemin du modèle
type_classifier = "bayesian" # Choisir entre "bayesian" et "kmeans"

if len(sys.argv) > 1:
    image_path = sys.argv[1]
    model_path = sys.argv[2]
    type_classifier = sys.argv[3]

if __name__ == "__main__":

    classifier = None

    if type_classifier == "bayesian":
        classifier = BayesianClassifier()
    elif type_classifier == "kmeans":
        classifier = KMeansClassifier()

    pipeline = ObjectDetectionPipeline(image_path)

    pipeline.load_model(model_path, classifier)

    pipeline.load_image()

    start_time = time.time()

    class_counts, detected_objects = pipeline.detect_and_classify_objects()

    end_time = time.time()

    print(f"Temps d'exécution: {end_time - start_time} secondes")

    print("Comptage des objets :", class_counts)

    counts_detected = 0
    if detected_objects is not None : counts_detected = len(detected_objects)
    print("Objets détectés :", counts_detected)

    pipeline.display_results(class_counts, detected_objects)

    # Vérité terrain (nombre d'objets par classe) de l'image
    true_counts = {
        "A": 64,
        "B": 4,
        "C": 30,
        "D": 34,
        "E": 137,
        "F": 7,
        "G": 16,
        "H": 16,
        "I": 65,
        "J": 5,
        "K": 0,
        "L": 33,
        "M": 30,
        "N": 59,
        "O": 65,
        "P": 45,
        "Q": 9,
        "R": 55,
        "S": 51,
        "T": 75,
        "U": 46,
        "V": 13,
        "W": 1,
        "X": 4,
        "Y": 11,
        "Z": 5,
        "0": 5,
        "1": 1,
        "2": 9,
        "3": 2,
        "4": 1,
        "5": 2,
        "6": 1,
        "7": 1,
        "8": 3,
        "9": 0,
    }

    evaluate_performance(class_counts, true_counts)