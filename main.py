from data_loader import read_data_from_folder
from data_manipulation import save_tif

from time import perf_counter
import os

# TODO: make masks
# TODO: prepare all data we have (yea, we have...)
# TODO: make A LOT of correlation img, data, etc.


if __name__ == '__main__':
    img = read_data_from_folder(os.path.abspath('data/head'))
    t0 = perf_counter()
    save_tif(img.flood_mask(), img_name='flood_3d_test')
    t1 = perf_counter()
    print(f"flood masks take {t1-t0} s to compute")
