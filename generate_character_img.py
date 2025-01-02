from PIL import Image, ImageDraw, ImageFont
import os


def generate_character_images(output_folder, font_path, font_size, image_size):
    characters = "abcdefghijklmnopqrstuvwxyz0123456789"

    # Charger la police
    font = ImageFont.truetype(font_path, font_size)

    for char in characters:
        # Identifier le dossier correspondant (lettre en majuscule)
        char_folder = os.path.join(output_folder, char.upper())
        os.makedirs(char_folder, exist_ok=True)

        # Créer une image blanche
        img = Image.new("RGB", image_size, color="white")
        draw = ImageDraw.Draw(img)

        # Calculer la position pour centrer le caractère
        text_width, text_height = draw.textbbox((0, 0), char, font=font)[2:4]
        position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)

        # Dessiner le caractère
        draw.text(position, char, fill="black", font=font)

        # Nommer le fichier selon majuscule ou minuscule
        if char.isupper():
            img_name = f"{char}_upper.png"
        elif char.islower():
            img_name = f"{char}_lower.png"
        else:
            img_name = f"{char}.png"  # Pour les chiffres

        # Sauvegarder l'image dans le dossier correspondant
        img_path = os.path.join(char_folder, img_name)
        img.save(img_path)


# Paramètres
output_folder = "output_characters"  # Dossier racine pour enregistrer les images
font_path = "arial.ttf"  # Chemin vers une police TrueType (remplacez par une police installée)
font_size = 64  # Taille de la police
image_size = (64, 64)  # Dimensions des images (pixels)

# Générer les images
os.makedirs(output_folder, exist_ok=True)
generate_character_images(output_folder, font_path, font_size, image_size)
