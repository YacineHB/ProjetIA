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

    try:
        # Assembler les lettres avec les lettre finisant avec un tiret du bas (ex : A_ avec A)
        for obj_name in list(counts.keys()):  # Crée une liste des clés pour éviter le problème
            if obj_name[-1] == "_":
                base_name = obj_name.split("_")[0]
                counts[base_name] = counts.get(base_name, 0) + counts.get(obj_name, 0)
                del counts[obj_name]  # Supprime la clé "_"

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
    except Exception as e:
        print(f"Erreur lors de l'évaluation des performances : {e}")

def enlarge_contour(dimension, top=0, bottom=0, left=0, right=0):
    """
    Agrandir un contour donné de quelques pixels dans les directions spécifiées.

    :param x: Coordonnée x du contour
    :param y: Coordonnée y du contour
    :param w: Largeur du contour
    :param h: Hauteur du contour
    :param top: Pixels à ajouter vers le haut
    :param bottom: Pixels à ajouter vers le bas
    :param left: Pixels à ajouter à gauche
    :param right: Pixels à ajouter à droite
    :return: Nouvelle boîte englobante agrandie (x, y, w, h)
    """
    x_new = max(0, dimension[0] - left)
    y_new = max(0, dimension[1] - top)
    w_new = dimension[2] + left + right
    h_new = dimension[3] + top + bottom
    return x_new, y_new, w_new, h_new