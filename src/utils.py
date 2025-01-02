from tensorflow.python.ops.metrics_impl import true_positives


def evaluate_performance(found_locations, counts):
    """
    Évaluer les performances du système de détection.
    :param found_locations: Dictionnaire contenant les positions des objets trouvés.
    :param counts: Nombre d'objets détectés par type.
    """
    # Implémentation simple pour évaluer la précision et le rappel
    # Par exemple, en comparant les résultats avec une vérité terrain (ici simulée)

    # Exemple : performance de précision / rappel fictifs pour chaque objet
    true_counts = {
        "A":64,
        "B":4,
        "C":30,
        "D":34,
        "E":137,
        "F":7,
        "G":16,
        "H":16,
        "I":65,
        "J":5,
        "K":0,
        "L":33,
        "M":30,
        "N":59,
        "O":65,
        "P":45,
        "Q":9,
        "R":55,
        "S":51,
        "T":75,
        "U":46,
        "V":13,
        "W":1,
        "X":4,
        "Y":11,
        "Z":5,
        "0":5,
        "1":1,
        "2":9,
        "3":2,
        "4":1,
        "5":2,
        "6":1,
        "7":1,
        "8":3,
        "9":0,
    }  # Exemple de vérité terrain
    precision = {}
    recall = {}
    find = {}
    true = {}


    for obj_name, detected_count in counts.items():
        true_count = true_counts.get(obj_name, 0)
        true[obj_name] = true_count
        find[obj_name] = detected_count

        precision[obj_name] = detected_count / (
                    detected_count + max(0, true_count - detected_count)) if detected_count > 0 else 0
        recall[obj_name] = detected_count / true_count if true_count > 0 else 0

    print("Précision et Rappel :")
    for obj_name in counts:
        print(f"{obj_name}: Précision={precision.get(obj_name, 0)}, Rappel={recall.get(obj_name, 0)}, Find={find.get(obj_name,0)}, Vrai={true.get(obj_name, 0)}")
