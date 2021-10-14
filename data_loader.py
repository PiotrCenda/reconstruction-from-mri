import os
import numpy as np
import io
from skimage import io

DATA_PATH = 'data'


def normalize(a):
    return (a - np.min(a)) / np.ptp(a)


def read_data():
    images = []
    for folder in os.listdir(DATA_PATH):
        #  tif only version
        img_dict = {'T1': None,
                    'T2': None}
        for key in img_dict.keys():
            img_path = os.path.join(DATA_PATH, folder, str(key + '.tif'))
            try:
                img_dict[key] = normalize(io.imread(img_path))
            except FileNotFoundError:
                print(key, "not found in", os.path.join(DATA_PATH, folder) + ". Will be set to None.")
                img_dict[key] = None

        images.append(ImageSequences(img_dict))

    if len(images) > 1:
        print("More than one data loaded. Returning as a list of ImageSequences classes. \n")
        return images
    else:
        print("Only one data loaded. Returning as a ImageSequences class. \n")
        return images[0]


class ImageSequences:

    def __init__(self, img_dict):
        self.__all = img_dict
        self.__t1 = img_dict['T1']
        self.__t2 = img_dict['T2']
        self.__middle = self.t1.shape[0] // 2

    def __copy__(self, data_dict=None):
        copy = ImageSequences(self.__all)
        return copy

    @property
    def t1(self):
        return self.__t1

    @property
    def t2(self):
        return self.__t2

    @property
    def middle(self):
        return self.__middle

    def thresh(self, seq='T1', val=0, val2=1):
        first_thresh = self.__all[seq] >= val
        sec_thresh = self.__all[seq] <= val2
        thresh = (first_thresh * sec_thresh).astype(int)
        del first_thresh
        del sec_thresh
        copy_dict = {'T1': self.__t1, 'T2': self.__t2, seq: thresh}
        return ImageSequences(copy_dict)

    def mask(self, seq='T1', val=0, val2=1):
        first_thresh = self.__all[seq] >= val
        sec_thresh = self.__all[seq] <= val2
        thresh = (first_thresh * sec_thresh).astype(int)
        thresh = thresh * self.__all[seq]
        del first_thresh
        del sec_thresh
        copy_dict = {'T1': self.__t1, 'T2': self.__t2, seq: thresh}
        return ImageSequences(copy_dict)
