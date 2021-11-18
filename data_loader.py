import os
import numpy as np
from skimage import io

from image import ImageSequences

DATA_PATH = 'data'


def normalize(a):
    """
    Simple normalization to 0-1
    """
    return (a - np.min(a)) / np.ptp(a)


def read_data_from_folder(folder_path):
    """
    ONLY .TIF VERSION

    Function loads T1 and T2 images from given folder as objects of ImageSequences class and then returns it.
    """
    img_dict = {'T1': None,
                'T2': None}

    for key in img_dict.keys():
        img_path = os.path.join(folder_path, str(key + '.tif'))
        try:
            img_dict[key] = normalize(io.imread(img_path))
        except FileNotFoundError:
            print(key, "not found in", folder_path + ". Will be set to None.")
            img_dict[key] = None

    print("Data from folder loaded. Returning as a ImageSequences class.\n")
    return ImageSequences(img_dict)


def read_all_data(data_path=DATA_PATH):
    """
    ONLY .TIF VERSION

    Function walks through "data" dictionary subfolders and loads T1 and T2 images as objects of ImageSequences class.
    Then returns list made of all these objects (if there is only one object, it returns it bare, without list).
    """
    images = []

    for folder in os.listdir(data_path):
        img_dict = {'T1': None,
                    'T2': None}
        for key in img_dict.keys():
            img_path = os.path.join(DATA_PATH, folder, str(key + '.tif'))
            try:
                img_dict[key] = normalize(io.imread(img_path))
            except FileNotFoundError:
                print(key, "not found in", os.path.join(DATA_PATH, folder) + ". Will be set to None.")
                img_dict[key] = None

        images.append(ImageSequences(img_dict))

    if len(images) > 1:
        print("More than one data loaded. Returning as a list of ImageSequences classes. \n")
        return images
    else:
        print("Only one data loaded. Returning as a ImageSequences class. \n")
        return images[0]
