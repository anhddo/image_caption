import argparse
import os
import numpy as np
from PIL import Image

def resize_image(image, size):
    """Resize an image to the given size."""
    m = np.min(image.size)
    im_size = np.floor(image.size / m * size)#tuple
    image = image.resize(im_size.astype(int), Image.ANTIALIAS)
    y, x = np.array(image.size) / 2
    image = image.crop((y - size/2, x - size / 2, y + size / 2 , x + size / 2))
    return image

def resize_images(image_dir, output_dir, size):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)
    for i, image in enumerate(images):
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img, size)
                img.save(os.path.join(output_dir, image), img.format)
        if i % 100 == 0:
            print ("[%d/%d] Resized the images and saved into '%s'."
                   %(i, num_images, output_dir))


if __name__ == '__main__':
    resize_images('data/Flickr8k_Dataset', 'data/resized256', 256)
