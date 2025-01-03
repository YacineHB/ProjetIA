from src.classifiers.bayesian import BayesianClassifier

if __name__ == "__main__":
    catalog_path = "data/catalogue"  # Données d'entraînement (sous-dossiers pour chaque classe)
    model_path = "models/bayesian_model.pth"  # Fichier pour sauvegarder le modèle

    classifier = BayesianClassifier()
    classifier.load_model(model_path)  # Charger le modèle existant (s'il existe)

    classifier.train(catalog_path)  # Ajouter ou mettre à jour les données du catalogue
    classifier.save_model(model_path)  # Sauvegarder le modèle après mise à jour

    classifier.visualize_model()  # Visualiser les résultats
