import os
import numpy as np

from data_loader import read_data_from_folder
from data_manipulation import save_tif, timer_block
from data_plotting import plot_3d_surface, plot_3d
from data_rigid_transform import auto_t1_t2_fitting

# parameters calculated by auto fitting function
params_auto = np.array([2.01109052e-04, 1.57808256e-06, 3.65095064e-05, 3.50697591e-04, 2.56535195e-04,
                        -2.36831914e-04, 9.40511337e-01, 9.38207923e-01, 1.00130253e+00])

# TODO: update soft tissues and bone masks


if __name__ == '__main__':
    img = read_data_from_folder(os.path.abspath('data/head'))
    # img.t2_rigid_transform(parameters=params_auto)
    save_tif(img.t1,  img_name="t1")
    save_tif(img.t2, img_name="t2_before")
    auto_t1_t2_fitting(img)
    save_tif(img.t2, img_name="t2_after")

    with timer_block("bones mask making"):
        bones = img.bones_mask()
        plot_3d(bones)
