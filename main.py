import os

from data_loader import read_data_from_folder
from data_manipulation import save_tif


# TODO: plotting etc. func
# TODO: prepare all data we have (yea, we have...)
# TODO: make A LOT of correlation img, data, etc.


if __name__ == '__main__':
    img = read_data_from_folder(os.path.abspath('data/head'))
    save_tif(img.t1, "tif_save_test", )
