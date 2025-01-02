from src.pipeline import ObjectDetectionPipeline
from src.utils import evaluate_performance
import time

if __name__ == "__main__":
    image_path = "data/plans/page.png"  # Image à traiter

    # Initialisation du pipeline de détection d'objets
    pipeline = ObjectDetectionPipeline(image_path)

    # Chargement du modèle
    pipeline.load_model("models/bayesian_model.pth")  # Modèle entraîné que vous avez sauvegardé

    # Chargement de l'image
    pipeline.load_image()

    # Mesure de la performance avant la détection
    start_time = time.time()

    # Détection et classification des objets dans l'image
    class_counts, detected_objects = pipeline.detect_and_classify_objects()

    # Mesure du temps d'exécution
    end_time = time.time()
    print(f"Temps d'exécution: {end_time - start_time} secondes")

    # Affichage des résultats
    print("Comptage des objets :", class_counts)
    # Affichage des objets détectés
    print("Objets détectés :", len(detected_objects))
    pipeline.display_results(class_counts, detected_objects)

    # Évaluation de la performance (précision, rappel, etc.)
    evaluate_performance(class_counts, class_counts)