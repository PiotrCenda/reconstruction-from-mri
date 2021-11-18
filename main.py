import matplotlib.pyplot as plt

from data_loader import read_data_from_folder
from data_manipulation import save_tif
import numpy as np
from data_rigid_transform import rigid_transform
import os


# TODO: make masks
# TODO: prepare all data we have (yea, we have...)
# TODO: make A LOT of correlation img, data, etc.


if __name__ == '__main__':
    img = read_data_from_folder(os.path.abspath('data/head'))
    # params = np.array([0, 0, 0, 0, 0, 0, 0.95, 0.96, 1])
    # trans_t2 = rigid_transform(img.t2>0.05, params)
    # slice = 87
    # f, ax = plt.subplots(2, 2)
    # plt.set_cmap('gray')
    # ax[0][0].imshow(np.invert(img.t1[slice]>0.05))
    # ax[0][0].set_title('t1')
    # ax[0][1].imshow(img.t2[slice]>0.05)
    # ax[0][1].set_title('t2')
    # ax[1][0].imshow(trans_t2[slice])
    # ax[1][0].set_title('t2 transformed')
    # ax[1][1].imshow(np.logical_xor(trans_t2>0.05,np.invert(img.t1[slice]>0.05))[slice])
    # ax[1][1].set_title('trans t2-t1')
    # plt.show()
    save_tif(np.logical_xor(img.background_mask(), img.flood_mask()), img_name='scaling_xor_test')

