import os
import numpy as np
from PIL import Image
from skimage import io
import re


def gif_maker(images_array, name=None, duration=100):
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

    images[0].save(path, save_all=True, append_images=images[1:], optimize=False, duration=duration, loop=0)
    del images


def image_folder_loader(path):
    '''
    Loads all images (as gray) to numpy array from folder under given path
    '''
    filenames = os.listdir(path)
    images_array = [io.imread((os.path.join(path, filename)), as_gray=True) for filename in filenames]
    return images_array


def save_img_array_to_tif(path):
    t1_filenames = [filename for filename in os.listdir(path) if re.search(r"t1", filename, re.I)
                    and not re.search(r"tirm", filename, re.I)]
    t2_filenames = [filename for filename in os.listdir(path) if re.search(r"t2", filename, re.I)
                    and not re.search(r"tirm", filename, re.I)]
    tirm_filenames = [filename for filename in os.listdir(path) if re.search(r"tirm", filename, re.I)]

    b16_files = set(filename for filename in os.listdir(path) if re.search(r"b16", filename, re.I))

    t1_images = [io.imread((os.path.join(path, filename)), as_gray=True) for filename in t1_filenames if filename not in b16_files]
    t2_images = [io.imread((os.path.join(path, filename)), as_gray=True) for filename in t2_filenames if filename not in b16_files]
    tirm_images = [io.imread((os.path.join(path, filename)), as_gray=True) for filename in tirm_filenames if filename not in b16_files]

    t1_images = [Image.fromarray(image.astype(np.uint8)).convert("P") for image in t1_images]
    t2_images = [Image.fromarray(image.astype(np.uint8)).convert("P") for image in t2_images]
    tirm_images = [Image.fromarray(image.astype(np.uint8)).convert("P") for image in tirm_images]

    folder_path = os.path.join("data", str(len(os.listdir("data"))))

    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    except OSError:
        print("Error: creating dir")

    t1_images[0].save(os.path.join(folder_path, "t1.tif"), save_all=True, append_images=t1_images[1:])
    t2_images[0].save(os.path.join(folder_path, "t2.tif"), save_all=True, append_images=t2_images[1:])
    tirm_images[0].save(os.path.join(folder_path, "tirm.tif"), save_all=True, append_images=tirm_images[1:])

    del t1_filenames
    del t2_filenames
    del tirm_filenames
    del b16_files
    del t1_images
    del t2_images
    del tirm_images
