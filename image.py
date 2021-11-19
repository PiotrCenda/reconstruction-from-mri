from skimage.segmentation import flood
from skimage.morphology import remove_small_holes, remove_small_objects, disk, closing, binary_erosion
import numpy as np
import scipy.ndimage as nd

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

    def shape(self):
        return self.__shape

    def t2_rigid_transform(self, parameters):
        self.__t2 = rigid_transform(self.__t2, parameters)

    def median(self):
        return np.array([nd.median_filter(img, footprint=disk(3)) for img in self.__t1]).astype(np.float64)

    @func_timer
    def background_mask(self):
        """
        based on t1 and t2 --> t2 gives problem with mask leaks and additionally t1 and t2 are shifted...
        """
        median_t1 = np.array([nd.median_filter(img, footprint=disk(2)) for img in self.__t1]).astype(np.float64)
        median_t2 = np.array([nd.median_filter(img, footprint=disk(2)) for img in self.__t2]).astype(np.float64)
        background_flood_t1 = np.array([flood(img, (0, 0), tolerance=0.05) for img in median_t1])
        background_flood_t2 = np.array([flood(img, (0, 0), tolerance=0.04) for img in median_t2])
        or_img = np.logical_and(background_flood_t1, background_flood_t2)
        remove_noise = np.array([remove_small_holes(img, area_threshold=300) for img in or_img])
        remove_noise_2 = np.array([remove_small_objects(img, min_size=300) for img in remove_noise])
        return np.array([closing(img, disk(5)) for img in remove_noise_2])

    @func_timer
    def soft_tissues(self):
        """
        ! experiment dark/light soft tissues and erosion etc.
        """
        median_t1 = np.array([nd.median_filter(img, footprint=disk(2)) for img in self.__t1]).astype(np.float64)
        median_t2 = np.array([nd.median_filter(img, footprint=disk(2)) for img in self.__t2]).astype(np.float64)

        thresh_t1_dark = np.logical_and(median_t1 <= 0.4, median_t1 >= 0.14)
        thresh_t1_light = median_t1 >= 0.4
        thresh_t1 = np.logical_or(thresh_t1_light, thresh_t1_dark)
        thresh_t2 = median_t2 >= 0.27

        remove_noise_t1 = np.array([remove_small_holes(img, area_threshold=20) for img in thresh_t1])
        remove_noise_t2 = np.array([remove_small_holes(img, area_threshold=20) for img in thresh_t2])

        sum_t1_t2 = np.logical_or(remove_noise_t1, remove_noise_t2)
        result = np.array([remove_small_objects(img, min_size=50) for img in sum_t1_t2])

        return result

    @func_timer
    def flood_mask(self):
        median = np.array([nd.median_filter(img, footprint=disk(3)) for img in self.__t2]).astype(np.float64)
        return flood(median, (0, 0, 0), tolerance=0.04)

    @func_timer
    def bones_mask(self):
        return remove_small_objects(np.logical_and(binary_erosion(binary_erosion(np.invert(
            self.background_mask()))), self.flood_mask()), min_size=40)
