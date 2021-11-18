from data_loader import read_data_from_folder
from data_manipulation import save_tif
from data_plotting import plot_3d

from time import perf_counter
import os
import numpy as np
from skimage.morphology import remove_small_objects

# TODO: make masks
# TODO: prepare all data we have (yea, we have...)
# TODO: make A LOT of correlation img, data, etc.


if __name__ == '__main__':
    img = read_data_from_folder(os.path.abspath('data/head'))
    data = remove_small_objects(np.logical_xor(img.flood_mask(), img.background_mask()), min_size=30)
    # save_tif(data, img_name='internal')

    t0 = perf_counter()
    plot_3d(data)
    t1 = perf_counter()
    print(f"plotting takes {t1-t0} s to compute")
