import os
import numpy as np
from PIL import Image


def gif_maker(images_array, name=None):
    '''
    Function which takes .tif array of images and saves it as .gif file under generic or given (as optional arg) name
    in results/gifs folder
    '''

    images = [Image.fromarray((image * 255).astype(np.uint8)).convert("P") for image in images_array]
    path = os.path.join("results", "gifs")

    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print("Error: creating dir")

    if name is None:
        name = str(len(os.listdir(path)))

    path = os.path.join(path, (name + ".gif"))

    images[0].save(path, save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)
