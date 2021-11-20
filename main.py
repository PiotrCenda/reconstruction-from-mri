from data_loader import read_data_from_folder
from data_manipulation import save_tif, timer_block
from data_plotting import plot_3d
from data_rigid_transform import register_image, rigid_transform
from data_manipulation import doce
import os
import numpy as np
import matplotlib.pyplot as plt

# TODO: make masks
# TODO: make automatic rigid transform (?)


if __name__ == '__main__':
    img = read_data_from_folder(os.path.abspath('data/head'))
    img_after_doce = doce(img=img.t1>0.05, command='25o75c')
    slice = 25
    f, ax = plt.subplots(2,1)
    ax[0].imshow((img.t1>0.1)[slice], cmap='gray')
    ax[1].imshow(img_after_doce[slice], cmap='gray')
    plt.show()
    # with timer_block('t2 to t1 registration'):
        # params = register_image(image_model=(img.t1>0.05).astype(np.int),
        #                         image_to_change=(img.t2>0.05).astype(np.int))

    # with timer_block('bones mask making'):
    #     bones = img.bones_mask()
    #
    # with timer_block('saving'):
    #     save_tif(bones, img_name='bones_mask_test')
    #     save_tif(img.background_mask(), img_name='background_mask')
    #     save_tif(img.flood_mask(), img_name='flood_mask')

    # plot_3d(bones)
