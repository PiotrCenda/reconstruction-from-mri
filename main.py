import os
import numpy as np

from data_loader import read_data_from_folder
from data_manipulation import save_tif, timer_block
from data_plotting import plot_3d_surface, plot_3d

# parameters calculated by auto fitting function
params_auto = np.array([1.51709147e+00, 1.18339327e+00, -1.17477642e-02, 7.45291130e-02, -1.40245845e+00,
                        4.35379594e-01, -2.87405872e-01, -4.27344509e+01, 7.68620321e-12, -3.70788805e-03,
                        -7.19916508e-02, -2.79734267e-03])

# TODO: update soft tissues and bone masks
# TODO: update automatic rigid transform
# TODO: add plots from marching_cubes function


if __name__ == '__main__':
    img = read_data_from_folder(os.path.abspath('data/head'))
    img.t2_rigid_transform(parameters=params_auto)

    with timer_block("bones mask making"):
        bones = img.bones_mask()
        plot_3d(bones)
