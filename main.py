from src.pipeline import ObjectDetectionPipeline
from src.utils import evaluate_performance
import time

if __name__ == "__main__":
    image_path = "data/plans/page.png"  # Image à traiter
    catalog_path = "data/catalogue"  # Dossier contenant les objets à rechercher

    # Initialisation du pipeline de détection d'objets
    pipeline = ObjectDetectionPipeline(image_path, catalog_path)

    # Chargement de l'image et du catalogue
    pipeline.load_image()
    pipeline.load_catalog()

    # Prétraitement de l'image
    processed_image = pipeline.preprocess_image()

    # Mesure de la performance avant la détection
    start_time = time.time()

    # Détection des objets dans l'image
    found_locations = pipeline.detect_objects(processed_image)

    # Comptage des objets détectés
    counts = pipeline.count_objects(found_locations)

    # Mesure du temps d'exécution
    end_time = time.time()
    print(f"Temps d'exécution: {end_time - start_time} secondes")

    # Affichage des résultats
    print("Comptage des objets :", counts)
    pipeline.display_results(found_locations)

    # Évaluation de la performance (précision, rappel, etc.)
    evaluate_performance(found_locations, counts)
