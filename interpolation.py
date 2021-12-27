import os
import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.util import img_as_ubyte
from skimage.exposure import adjust_log, adjust_sigmoid

from data_manipulation import save_tif


mkdir_error_message = "Error: creating dir"
directory = 'temp'


def interpolate(img):
    scale_z_to_y(img)
    interpolated = zy_to_tif()
    # comment to cephalo
    # interpolated[interpolated >= 1] = 255
    # interpolated[interpolated < 1] = 0
    return img_as_ubyte(interpolated)


def scale_z_to_y(img):
    aspect_ratio = img.shape[1]/img.shape[0]
    my_dpi = int(img.shape[1]/8)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 8)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print(mkdir_error_message)

    print(f"Saving scaled image slices to \"{directory}\" folder...")

    for img_slice in tqdm(range(img.shape[2]), desc="Saving: "):
        ax.imshow(img[:, img_slice, :], interpolation='bicubic', cmap='gray', aspect=aspect_ratio)
        fig.savefig(os.path.join(directory, (str(img_slice) + '.png')), dpi=my_dpi)
        ax.clear()


def zy_to_tif():
    img = []
    img_paths = [os.path.join(directory, f) for f in os.listdir(directory)]
    img_paths.sort(key=len)

    print(f"Loading scaled image slices from \"{directory}\" folder...")

    for path in tqdm(img_paths, desc="Loading: "):
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img.append(im)

    img = np.array(img)
    img = np.swapaxes(img, 0, 1)

    shutil.rmtree(directory, ignore_errors=True)

    return img


def show_xyz(img):
    z_slice = img.shape[0] // 2
    y_slice = img.shape[1] // 2
    x_slice = img.shape[2] // 2

    f, ax = plt.subplots(1, 3)
    ax[0].imshow(img[z_slice, :, :], cmap='gray')
    ax[1].imshow(img[::-1, y_slice, :], cmap='gray')
    ax[2].imshow(img[::-1, :, x_slice], cmap='gray')
    plt.show()


def normalize(a):
    """
    Simple normalization to 0-1
    """
    return (a - np.min(a)) / np.ptp(a)


def cephalo(img, soft):
    z_slice = np.zeros(img[0, :, :].shape)
    y_slice = np.zeros(img[:, 0, :].shape)
    x_slice = np.zeros(img[:, :, 0].shape)

    z_soft = np.zeros(img[0, :, :].shape)
    y_soft = np.zeros(img[:, 0, :].shape)
    x_soft = np.zeros(img[:, :, 0].shape)

    for i in range(img.shape[0]):
        z_slice += img[i, :, :]
    for i in range(img.shape[1]):
        y_slice += img[::-1, i, :]
    for i in range(img.shape[2]):
        x_slice += img[::-1, :, i]

    for i in range(soft.shape[0]):
        z_soft += soft[i, :, :]
    for i in range(soft.shape[1]):
        y_soft += soft[::-1, i, :]
    for i in range(soft.shape[2]):
        x_soft += soft[::-1, :, i]

    x_slice = normalize(normalize(np.sqrt(x_slice)) + 0.25 * normalize(np.sqrt(x_soft)))
    y_slice = normalize(normalize(np.sqrt(y_slice)) + 0.25 * normalize(np.sqrt(y_soft)))
    z_slice = normalize(normalize(np.sqrt(z_slice)) + 0.25 * normalize(np.sqrt(z_soft)))

    save_tif(x_slice, img_name="x_slice", folder="cephalometry")
    save_tif(y_slice, img_name="y_slice", folder="cephalometry")
    save_tif(z_slice, img_name="z_slice", folder="cephalometry")
