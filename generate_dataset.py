import os
import random
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import zipfile
import io
import datetime
import math

# import assitant module
from shapes import get_random_shape_function, SHAPE_IDS

def color_distance(c1, c2):
    """calculate the Euclidian distance between colors ins RGB space."""
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(c1, c2)]))

def random_color():
    """returns a tuple from random RGB."""
    return tuple(np.random.randint(0, 256, size=3))

def generate_and_add_to_zip(i, images_zip, labels_zip, img_size, color_mode='random'):
    """Generate a random shape image and add it to the dataset archives."""
    CONTRAST_THRESHOLD = 120
    pure_palette = [
        (255, 0, 0),    # red
        (0, 255, 0),    # green
        (0, 0, 255),    # blue
        (0, 0, 0),      # black
    ]

    if color_mode == 'pure':
        background_color, shape_color = random.sample(pure_palette, 2)
    else:
        background_color = random_color()
        shape_color = random_color()
        while color_distance(background_color, shape_color) < CONTRAST_THRESHOLD:
            shape_color = random_color()

    img = Image.new("RGB", (img_size, img_size), background_color)
    draw = ImageDraw.Draw(img)

    draw_func, shape_name = get_random_shape_function()

    actual_shape_name = draw_func(draw, img_size, shape_color, background_color=background_color)

    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_filename = f"{i:06}.png"
    images_zip.writestr(img_filename, img_buffer.getvalue())

    shape_id = SHAPE_IDS[actual_shape_name]
    label_content = f"{shape_id}\n"
    label_filename = f"{i:06}.txt"
    labels_zip.writestr(label_filename, label_content)

def generate_data(num_images, img_size, color_mode='random'):
    """Generate a dataset and return the output directory path."""
    valid_modes = {'random', 'pure'}
    if color_mode not in valid_modes:
        raise ValueError(f"color_mode must be one of {sorted(valid_modes)}, got '{color_mode}'.")

    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M")
    mode_suffix = f"_{color_mode}" if color_mode != 'random' else ''
    output_parent_dir = f"{timestamp}_{num_images}imgs_{img_size}x{img_size}{mode_suffix}"
    os.makedirs(output_parent_dir, exist_ok=True)

    output_images_path = os.path.join(output_parent_dir, "images.zip")
    output_labels_path = os.path.join(output_parent_dir, "labels.zip")

    print(f"Saved in: '{output_parent_dir}' (color_mode={color_mode})")

    with zipfile.ZipFile(output_images_path, 'w', zipfile.ZIP_DEFLATED) as images_zip:
        with zipfile.ZipFile(output_labels_path, 'w', zipfile.ZIP_DEFLATED) as labels_zip:
            for i in tqdm(range(num_images), desc=f"Generating ({color_mode})"):
                generate_and_add_to_zip(i, images_zip, labels_zip, img_size, color_mode=color_mode)

    print()
    print("Sucess to generating shapes!")
    with open(os.path.join(output_parent_dir, 'shape_ids.txt'), 'w') as f:
        for name, idx in SHAPE_IDS.items():
            f.write(f"{idx}: {name}\n")

    return output_parent_dir


if __name__ == "__main__":
    generate_data(num_images=12000, img_size=28)
