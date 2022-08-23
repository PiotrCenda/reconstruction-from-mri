import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from data_loader import read_tif
from data_manipulation import save_tif, timer_block
from pantomography import rotate, generate_pantomography

# parameters calculated by auto fitting function
# params_auto_old = np.array([2.01109052e-04, 1.57808256e-06, 3.65095064e-05, 3.50697591e-04, 2.56535195e-04,
#                         -2.36831914e-04, 9.40511337e-01, 9.38207923e-01, 1.00130253e+00])
params_auto = np.array([2.77076163e-03, -7.52885706e-03, 7.03755373e-04, 3.79097329e-01, -2.86304089e-03,
                        6.44776348e-01, 9.39824479e-01, 9.40039058e-01, 1.00105912e+00])


if __name__ == '__main__':
    bone_tissues = read_tif(os.path.abspath('data/tissues/bones_mask_interpolated.tif'))
    soft_tissues = read_tif(os.path.abspath('data/tissues/soft_mask_interpolated.tif'))

    with timer_block("pantomography reconstruction"):
        bone_tissues = rotate(bone_tissues)
        pantomography = generate_pantomography(bone_tissues)
        
        plt.figure()
        plt.imshow(pantomography, cmap='gray')
        folder_path = Path("results/pantomography")
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(str(folder_path), "pantomography.png")
        plt.imsave(file_path, pantomography, cmap='gray', dpi=300)
        plt.close()

