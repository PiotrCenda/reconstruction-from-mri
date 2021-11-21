from skimage.segmentation import flood
from skimage.morphology import remove_small_holes, remove_small_objects, disk, closing
from skimage.measure import label, regionprops
import scipy.ndimage as nd
import numpy as np

from data_rigid_transform import rigid_transform
from data_manipulation import func_timer, doce, save_tif
from data_plotting import plot_3d


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
        background_flood_t1 = np.array([flood(img, (0, 0), tolerance=0.05) for img in self.__t1])
        background_flood_t2 = np.array([flood(img, (0, 0), tolerance=0.05) for img in self.__t2])
        and_img = np.logical_and(background_flood_t1, background_flood_t2)

        background_size = max([region.area for region in regionprops(label(and_img, connectivity=3))])
        remove_objects = remove_small_objects(and_img, min_size=(background_size - 1), connectivity=3)
        closed = np.array([closing(img, disk(3)) for img in remove_objects])
        not_background = max([region.area for region in regionprops(label(np.logical_not(closed), connectivity=3))])
        remove_holes = remove_small_holes(closed, area_threshold=(not_background - 1), connectivity=3)
        # dilations = np.array(doce(remove_objects, "3d")).astype(np.bool_)

        return remove_holes

    @func_timer
    def soft_tissues(self):
        """
        ! experiment dark/light soft tissues and erosion etc.
        """
        median_t1 = np.array([nd.median_filter(img, footprint=disk(2)) for img in self.__t1]).astype(np.float64)
        median_t2 = np.array([nd.median_filter(img, footprint=disk(2)) for img in self.__t2]).astype(np.float64)

        thresh_t1_dark = np.logical_and(median_t1 <= 0.4, median_t1 >= 0.13)
        thresh_t1_light = median_t1 >= 0.4
        thresh_t1 = np.logical_or(thresh_t1_light, thresh_t1_dark)
        thresh_t2 = median_t2 >= 0.3

        sum_t1_t2 = np.logical_or(thresh_t1, thresh_t2)
        result = np.array([remove_small_objects(img, min_size=50) for img in sum_t1_t2])

        return result

    @func_timer
    def flood_mask(self):
        median_t1 = np.array([nd.median_filter(img, footprint=disk(2)) for img in self.__t1]).astype(np.float64)
        median_t2 = np.array([nd.median_filter(img, footprint=disk(2)) for img in self.__t2]).astype(np.float64)

        flood_mask_t1 = flood(median_t1, (0, 0, 0), tolerance=0.07)
        flood_mask_t2 = flood(median_t2, (0, 0, 0), tolerance=0.07)

        result = np.logical_or(flood_mask_t1, flood_mask_t2)

        return result

    @func_timer
    def bones_mask(self):
        # check logic in this function !!!
        no_background = doce(np.logical_not(self.background_mask()), "10e")
        internal_flood = np.logical_and(no_background, self.flood_mask())
        no_soft_tissues = np.logical_and(no_background, np.logical_not(self.soft_tissues()))

        return remove_small_objects(np.logical_and(internal_flood, no_soft_tissues), min_size=30)
