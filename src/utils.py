def evaluate_performance(counts, true_counts):
    """
    Évaluer les performances du système de détection.
    :param counts: Nombre d'objets détectés par type.
    """

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
        print(f"Précision moyenne : {sum(precision.values()) / len(precision)}, Rappel moyen : {sum(recall.values()) / len(recall)}")
        #Le rappel
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