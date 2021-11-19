from data_loader import read_data_from_folder
from data_manipulation import save_tif, timer_block
from data_plotting import plot_3d

import os
import numpy as np

# TODO: make masks
# TODO: make automatic rigid transform (?)


if __name__ == '__main__':
    img = read_data_from_folder(os.path.abspath('data/head'))
    img.t2_rigid_transform(parameters=np.array([0, 0, 0, 0, 0, 0, 0.95, 0.96, 1]))

    # with timer_block("soft tissues mask"):
    #     soft_tissue = img.soft_tissues()
    #     save_tif(soft_tissue, img_name="soft_tissue_mask")
    #     plot_3d(soft_tissue)

    with timer_block('bones mask making'):
        bones = img.bones_mask()
        save_tif(bones, img_name="bones_mask")
        plot_3d(bones)
