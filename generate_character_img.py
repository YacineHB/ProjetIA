from PIL import Image, ImageDraw, ImageFont
import os


def generate_character_images(output_folder, font_path, font_size, image_size):
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

    # Charger la police
    font = ImageFont.truetype(font_path, font_size)

    for char in characters:
        # Identifier le dossier correspondant (lettre en majuscule)

        if char.isupper():
            name = f"{char}_"
        else:
            name = f"{char.upper()}"


        char_folder = os.path.join(output_folder, name)
        os.makedirs(char_folder, exist_ok=True)

        # Créer une image blanche
        img = Image.new("RGB", image_size, color="white")
        draw = ImageDraw.Draw(img)

        # Calculer la position pour centrer le caractère
        text_width, text_height = draw.textbbox((0, 0), char, font=font)[2:4]
        position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2 - 2)

        # Dessiner le caractère
        draw.text(position, char, fill="black", font=font)

        img_name = f"{char}.png"

        # Sauvegarder l'image dans le dossier correspondant
        img_path = os.path.join(char_folder, img_name)
        img.save(img_path)


# Paramètres
output_folder = "output_characters"  # Dossier racine pour enregistrer les images
#Cambria font
font_path = "C:/Windows/Fonts/cambriab.ttf"  # Chemin de la police TrueType
font_size = 28  # Taille de la police
image_size = (30, 30)  # Dimensions des images (pixels)

# Générer les images
os.makedirs(output_folder, exist_ok=True)
generate_character_images(output_folder, font_path, font_size, image_size)
