def evaluate_performance(found_locations, counts):
    """
    Évaluer les performances du système de détection.
    :param found_locations: Dictionnaire contenant les positions des objets trouvés.
    :param counts: Nombre d'objets détectés par type.
    """
    # Implémentation simple pour évaluer la précision et le rappel
    # Par exemple, en comparant les résultats avec une vérité terrain (ici simulée)

    # Exemple : performance de précision / rappel fictifs pour chaque objet
    true_counts = {'a.png': 5, 'b.png': 3}  # Exemple de vérité terrain
    precision = {}
    recall = {}

    for obj_name, detected_count in counts.items():
        true_count = true_counts.get(obj_name, 0)
        precision[obj_name] = detected_count / (
                    detected_count + max(0, true_count - detected_count)) if detected_count > 0 else 0
        recall[obj_name] = detected_count / true_count if true_count > 0 else 0

    print("Précision et Rappel :")
    for obj_name in counts:
        print(f"{obj_name}: Précision={precision.get(obj_name, 0)}, Rappel={recall.get(obj_name, 0)}")
