import os
import re
import numpy as np
from PIL import Image
from time import perf_counter
from datetime import timedelta
from skimage import io, img_as_ubyte
from contextlib import contextmanager
from skimage.morphology import binary_erosion, binary_closing, binary_opening, binary_dilation

mkdir_error_message = "Error: creating dir"


def doce(img, command: str):
    """
    :param img: image to make operation on
    :param command: example "e5c3o1d" - 1 erosion 5 closing 3 openings 1 dilatation
    :return: transfomred image
    """
    command_dict = {'d': binary_dilation,
                    'o': binary_opening,
                    'c': binary_closing,
                    'e': binary_erosion}
    command = command.lower()
    command_len = len(command)
    current_element = 0
    img = img.astype(int)

    while current_element < command_len:
        if command[current_element].isnumeric():
            number_iterator = 0
            while command[current_element+number_iterator].isnumeric():
                number_iterator += 1
            print(command[current_element: current_element + number_iterator],
                  command_dict[command[current_element + number_iterator]].__name__)
            for _ in range(int(command[current_element: current_element + number_iterator])):
                img = command_dict[command[current_element + number_iterator]](img)
            current_element += number_iterator
        else:
            print(command_dict[command[current_element]].__name__)
            img = command_dict[command[current_element]](img)
        current_element += 1

    return img.astype(np.uint8)


def func_timer(function):
    """
    decorator for function execution time measuring
    """
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        func_return = function(*args, *kwargs)
        time_passed = timedelta(seconds=perf_counter() - start_time)
        func_name = function.__name__
        print(f"Function {func_name} took {time_passed} to complete.")
        return func_return
    return wrapper


@contextmanager
def timer_block(name: str):
    """
    context manager for block of code execution time measuring
    """
    start_time = perf_counter()
    yield
    time_passed = timedelta(seconds=perf_counter() - start_time)
    print(f"\nBlock named: \"{name}\" took {time_passed} to complete.\n")


def gif_maker(images_array, name=None, duration=100):
    """
    Function which takes .tif array of images and saves it as .gif file under generic or given (as optional arg) name
    in results/gifs folder.
    """

    images = [Image.fromarray((image * 255).astype(np.uint8)).convert("P") for image in images_array]
    path = os.path.join("results", "gifs")

    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(mkdir_error_message)

    if name is None:
        name = str(len(os.listdir(path)))

    path = os.path.join(path, (name + ".gif"))

    images[0].save(path, save_all=True, append_images=images[1:], optimize=False, duration=duration, loop=0)
    del images


def image_folder_loader(path: str):
    """
    Loads all images (as gray) to numpy array from folder under given path.
    """
    filenames = os.listdir(path)
    images_array = [io.imread((os.path.join(path, filename)), as_gray=True) for filename in filenames]
    return images_array


def save_img_array_to_tif(path: str):
    """
    Loads all images (as gray) to numpy array from folder under given path and then saves all found modalities
    as .tif files in "data" folder.
    """
    t1_filenames = [filename for filename in os.listdir(path) if re.search(r"t1", filename, re.I)
                    and not re.search(r"tirm", filename, re.I)]
    t2_filenames = [filename for filename in os.listdir(path) if re.search(r"t2", filename, re.I)
                    and not re.search(r"tirm", filename, re.I)]
    tirm_filenames = [filename for filename in os.listdir(path) if re.search(r"tirm", filename, re.I)]

    b16_files = set(filename for filename in os.listdir(path) if re.search(r"b16", filename, re.I))

    t1_images = [io.imread((os.path.join(path, filename)), as_gray=True) for filename in t1_filenames
                 if filename not in b16_files]
    t2_images = [io.imread((os.path.join(path, filename)), as_gray=True) for filename in t2_filenames if filename
                 not in b16_files]
    tirm_images = [io.imread((os.path.join(path, filename)), as_gray=True) for filename in tirm_filenames if filename
                   not in b16_files]

    t1_images = [Image.fromarray(image.astype(np.uint8)).convert("P") for image in t1_images]
    t2_images = [Image.fromarray(image.astype(np.uint8)).convert("P") for image in t2_images]
    tirm_images = [Image.fromarray(image.astype(np.uint8)).convert("P") for image in tirm_images]

    folder_path = os.path.join("data", str(len(os.listdir("data"))))

    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    except OSError:
        print(mkdir_error_message)

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


def save_tif(img, img_name=None, folder='test_masks'):
    """
    Saves .tif image as file in "results" folder.
    """
    folder_path = os.path.join("results", folder)

    if img_name is None:
        img_name = str(len(os.listdir(folder_path)))

    img_path = os.path.join(folder_path, str(img_name + ".tif"))

    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    except OSError:
        print(mkdir_error_message)

    io.imsave(img_path, img_as_ubyte(img))
