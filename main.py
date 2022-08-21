import os
import numpy as np

from data_loader import read_data_from_folder
from data_manipulation import save_tif, timer_block
from data_plotting import plot_3d
from interpolation import cephalo, interpolate

# parameters calculated by auto fitting function
# params_auto_old = np.array([2.01109052e-04, 1.57808256e-06, 3.65095064e-05, 3.50697591e-04, 2.56535195e-04,
#                         -2.36831914e-04, 9.40511337e-01, 9.38207923e-01, 1.00130253e+00])
params_auto = np.array([2.77076163e-03, -7.52885706e-03, 7.03755373e-04, 3.79097329e-01, -2.86304089e-03,
                        6.44776348e-01, 9.39824479e-01, 9.40039058e-01, 1.00105912e+00])


if __name__ == '__main__':
    img = read_data_from_folder(os.path.abspath('data/head'))
    img.t2_rigid_transform(parameters=params_auto)

    with timer_block("cephalometry reconstruction"):
        bones = img.bones_mask()
        # save_tif(bones, img_name="bones_mask", folder="tests")
        soft = img.soft_tissues()
        # save_tif(soft, img_name="soft_mask", folder="tests")

        soft_interpolated = interpolate(soft)
        save_tif(soft_interpolated, img_name="soft_mask_interpolated", folder="tests")
        bones_interpolated = interpolate(bones)
        save_tif(bones_interpolated, img_name='bones_mask_interpolated', folder='tests')

        cephalo(bones_interpolated, soft_interpolated)
