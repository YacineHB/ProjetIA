from src.classifiers.bayesian import BayesianClassifier
from src.classifiers.kmeans import KMeansClassifier
from src.pipeline import ObjectDetectionPipeline
from src.utils import evaluate_performance
import time

image_path = "data/plans/page.png" # Choisir le chemin de l'image
model_path = "models/bayesian_model.pth" # Choisir le chemin du modèle
type_classifier = "bayesian" # Choisir entre "bayesian" et "kmeans"

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

    evaluate_performance(class_counts, class_counts)