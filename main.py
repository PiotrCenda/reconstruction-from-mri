from data_loader import read_data_from_folder
from data_manipulation import save_tif
from data_plotting import plot_3d

import os
import numpy as np

# TODO: make masks
# TODO: prepare all data we have (yea, we have...)
# TODO: make A LOT of correlation img, data, etc.


if __name__ == '__main__':
    img = read_data_from_folder(os.path.abspath('data/head'))
    img.t2_rigid_transform(parameters=np.array([0, 0, 0, 0, 0, 0, 0.95, 0.96, 1]))

    bones = img.bones_with_air()
    save_tif(bones, img_name='scaling_xor_test')
    save_tif(img.background_mask(), img_name='background_mask')
    save_tif(img.flood_mask(), img_name='flood_mask')
    plot_3d(bones)
