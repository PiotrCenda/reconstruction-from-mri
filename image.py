import numpy as np
from skimage.segmentation import flood
from skimage.morphology import remove_small_holes, remove_small_objects, disk, closing, dilation
from skimage.measure import label, regionprops
from skimage.filters.rank import mean_bilateral
from skimage.util import img_as_ubyte
import multiprocessing as mp
from multiprocessing.pool import ThreadPool

from data_rigid_transform import rigid_transform
from data_manipulation import func_timer


class ImageSequences:
    """
    Class where T1 and T2 sequences are stored. Additionally there are functions which thresholds, masks and segments
    objects on images.
    """

    def __init__(self, img_dict):
        self.__all = img_dict
        self.__t1 = img_dict['T1']
        self.__t2 = img_dict['T2']
        self.__shape = img_dict['T1'][0].shape

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
    def shape(self):
        return self.__shape

    def t2_rigid_transform(self, parameters):
        self.__t2 = rigid_transform(self.__t2, parameters)

    @func_timer
    def background_mask(self):
        with mp.Pool(processes=mp.cpu_count()) as pool:
            p1 = pool.map_async(flood_wrap, [image for image in self.__t1])
            p2 = pool.map_async(flood_wrap, [image for image in self.__t2])

            and_img = np.logical_and(p1.get(), p2.get())

            background_size = max([region.area for region in regionprops(label(and_img, connectivity=3))])
            remove_objects = remove_small_objects(and_img, min_size=(background_size - 1), connectivity=3)

            dilated = pool.map(dilate_wrap, [image for image in remove_objects])
            closed = pool.map(closing_wrap, [image for image in dilated])

            not_background = max([region.area for region in regionprops(label(np.logical_not(closed), connectivity=3))])
            remove_holes = remove_small_holes(np.array(closed), area_threshold=(not_background - 1), connectivity=3)

        return remove_holes

    @func_timer
    def soft_tissues(self):
        """
        update filter and noise reduction
        """
        with mp.Pool(processes=mp.cpu_count()) as pool:
            p1 = pool.map_async(mean_bilateral_wrap, [image for image in self.__t1])
            p2 = pool.map_async(mean_bilateral_wrap, [image for image in self.__t2])
            t1 = p1.get()
            t2 = p2.get()

        t1 = np.array(((t1 - np.min(t1)) / np.ptp(t1))).astype(np.float64)
        t2 = np.array(((t2 - np.min(t2)) / np.ptp(t2))).astype(np.float64)

        thresh_t1 = t1 >= 0.1
        thresh_t2 = t2 >= 0.14

        result = np.logical_or(thresh_t1, thresh_t2)

        return result

    @func_timer
    def bones_mask(self):
        """
        remove sinuses and air, upgrade multiprocessing
        """
        with ThreadPool(processes=mp.cpu_count()) as pool:
            p1 = pool.apply_async(self.background_mask)
            p2 = pool.apply_async(self.soft_tissues)

            no_soft_tissues = np.logical_not(np.logical_or(p1.get(), p2.get()))

        skull_size = max([region.area for region in regionprops(label(no_soft_tissues, connectivity=3))])
        result = remove_small_objects(no_soft_tissues, min_size=(skull_size - 1), connectivity=3)

        return result


def mean_bilateral_wrap(img):
    return mean_bilateral(img_as_ubyte(img), disk(2))


def flood_wrap(img):
    return flood(img, seed_point=(0, 0), tolerance=0.05)


def dilate_wrap(img):
    return dilation(img, disk(10))


def closing_wrap(img):
    return closing(img, disk(5))
