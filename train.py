from src.classifiers.bayesian import BayesianClassifier
from src.classifiers.kmeans import KMeansClassifier
import sys

catalog_path = "data/catalogue" # Choisir le chemin du catalogue
model_path = "models/kmeans_model.pth" # Choisir le chemin du modèle
classifier_type = "kmeans" # Choisir entre "bayesian" et "kmeans"

if len(sys.argv) > 1:
    catalog_path = sys.argv[1]
    model_path = sys.argv[2]
    classifier_type = sys.argv[3]

if __name__ == "__main__":

    classifier = None

    if classifier_type == "bayesian":
        classifier = BayesianClassifier()
    elif classifier_type == "kmeans":
        classifier = KMeansClassifier()

    try:
        classifier.load_model(model_path)
        print(f"Modèle chargé depuis {model_path}.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}. Un nouveau modèle sera créé.")

    print("Entraînement du modèle en cours...")
    classifier.train(catalog_path)
    print("Entraînement terminé.")

    print(f"Sauvegarde du modèle dans {model_path}...")
    classifier.save_model(model_path)

    print("Visualisation des résultats...")
    classifier.visualize_model()
    print("Visualisation terminée.")