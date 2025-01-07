from src.classifiers.bayesian import BayesianClassifier
from src.classifiers.kmean import KMeansClassifier

if __name__ == "__main__":
    catalog_path = "data/catalogue"  # Données d'entraînement (sous-dossiers pour chaque classe)
    model_path = "models/bayesian_model.pth"  # Fichier pour sauvegarder le modèle entraîné

    classifier = BayesianClassifier() # Créer un classifieur bayésien ou kmeans

    # Tenter de charger le modèle existant, si il existe
    try:
        classifier.load_model(model_path)  # Charger le modèle existant
        print(f"Modèle chargé depuis {model_path}.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}. Un nouveau modèle sera créé.")

    # Entraîner le modèle avec les données du catalogue
    print("Entraînement du modèle en cours...")
    classifier.train(catalog_path)  # Ajouter ou mettre à jour les données du catalogue
    print("Entraînement terminé.")

    # Sauvegarder le modèle après mise à jour
    print(f"Sauvegarde du modèle dans {model_path}...")
    classifier.save_model(model_path)  # Sauvegarder le modèle après mise à jour

    # Visualiser les résultats (centroïdes des clusters)
    print("Visualisation des résultats...")
    classifier.visualize_model()  # Visualiser les résultats des clusters
    print("Visualisation terminée.")