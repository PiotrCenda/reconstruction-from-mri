import numpy as np
from skimage.segmentation import flood
from skimage.morphology import remove_small_holes, remove_small_objects, disk, closing, dilation
from skimage.measure import label, regionprops
from skimage.filters.rank import mean_bilateral
from skimage.util import img_as_ubyte
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from itertools import repeat

from data_rigid_transform import rigid_transform
from data_manipulation import func_timer, save_tif


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
            p1 = pool.map_async(mean_bilateral_wrap, [image for image in self.__t1])
            p2 = pool.map_async(mean_bilateral_wrap, [image for image in self.__t2])

            t1 = p1.get()
            t2 = p2.get()

            t1 = np.array(((t1 - np.min(t1)) / np.ptp(t1))).astype(np.float64)
            t2 = np.array(((t2 - np.min(t2)) / np.ptp(t2))).astype(np.float64)

            p1_2 = pool.map_async(flood_wrap, [image for image in t1])
            p2_2 = pool.map_async(flood_wrap, [image for image in t2])

            save_tif(np.array(p1.get()), img_name="bilateral_1", folder="masks")
            save_tif(np.array(p2.get()), img_name="bilateral_2", folder="masks")

            and_img = np.logical_and(p1_2.get(), p2_2.get())

            background_size = max([region.area for region in regionprops(label(and_img, connectivity=3))])
            remove_objects = remove_small_objects(and_img, min_size=(background_size - 1), connectivity=3)

            dilated = pool.starmap(dilation, zip(remove_objects, repeat(disk(7))))
            closed = pool.starmap(closing, zip(dilated, repeat(disk(5))))
            dilated2 = pool.starmap(dilation, zip(closed, repeat(disk(5))))

            not_background = max([region.area for region in regionprops(label(np.logical_not(dilated2), connectivity=3))])
            remove_holes = remove_small_holes(np.array(closed), area_threshold=(not_background - 1), connectivity=3)

        return remove_holes

    @func_timer
    def soft_tissues(self):
        """
        update filter and noise reduction
        """
        with mp.Pool(processes=mp.cpu_count()) as pool:
            p1 = pool.map_async(mean_bilateral_wrap2, [image for image in self.__t1])
            p2 = pool.map_async(mean_bilateral_wrap2, [image for image in self.__t2])
            t1 = p1.get()
            t2 = p2.get()

            t1 = np.array(((t1 - np.min(t1)) / np.ptp(t1))).astype(np.float64)
            t2 = np.array(((t2 - np.min(t2)) / np.ptp(t2))).astype(np.float64)

            thresh_t1 = t1 >= 0.1
            thresh_t2 = t2 >= 0.18

            p1_2 = pool.map_async(remove_wrap, [image for image in thresh_t1])
            p2_2 = pool.map_async(remove_wrap, [image for image in thresh_t2])
            t1 = np.array(p1_2.get()).astype(np.float64)
            t2 = np.array(p2_2.get()).astype(np.float64)

        save_tif(t1, img_name="thresh_t1", folder="masks")
        save_tif(t2, img_name="thresh_t2", folder="masks")

        result = np.logical_or(t1, t2)

        return result

    @func_timer
    def bones_mask(self):
        """
        remove sinuses and air, upgrade multiprocessing
        """
        with ThreadPool(processes=mp.cpu_count()) as pool:
            background = pool.apply_async(self.background_mask)
            soft = pool.apply_async(self.soft_tissues)

            no_soft_tissues = np.logical_not(np.logical_or(background.get(), soft.get()))

        remove = np.array([remove_small_objects(img, min_size=25) for img in no_soft_tissues])
        skull_size = max([region.area for region in regionprops(label(remove, connectivity=3))])
        result = remove_small_objects(remove, min_size=(skull_size - 1), connectivity=3)

        return result


def remove_wrap(img):
    img = remove_small_holes(img, area_threshold=10)
    return remove_small_objects(img, min_size=30)


def mean_bilateral_wrap(img):
    return mean_bilateral(img_as_ubyte(img), disk(7))


def mean_bilateral_wrap2(img):
    return mean_bilateral(img_as_ubyte(img), disk(3))


def flood_wrap(img):
    return flood(img, seed_point=(0, 0), tolerance=0.06)
