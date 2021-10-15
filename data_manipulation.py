import os
import numpy as np
from PIL import Image
from skimage import io


def gif_maker(images_array, name=None, time=100):
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

    images[0].save(path, save_all=True, append_images=images[1:], optimize=False, duration=time, loop=0)


def image_folder_loader(path):
    '''
    Loads all images (as gray) to numpy array from folder under given path
    '''
    filenames = os.listdir(path)
    images_array = [io.imread((os.path.join(path, filename)), as_gray=True) for filename in filenames]
    return images_array

