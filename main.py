import os

from data_loader import read_data_from_folder
from data_manipulation import *
from data_plotting import *


# TODO: plotting etc. func
# TODO: prepare all data we have (yea, we have...)
# TODO: make A LOT of correlation img, data, etc.


if __name__ == '__main__':
    img = read_data_from_folder(os.path.abspath('data/head'))
    img.background_mask()
