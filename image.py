from skimage.segmentation import flood, flood_fill
from skimage.morphology import remove_small_holes, remove_small_objects, disk, closing
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt


# TODO: add more masks, start to correlate them and modalities into process of mask making


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

    def median(self):
        median = np.array([nd.median_filter(img, footprint=disk(3)) for img in self.__t1]).astype(np.float64)
        return median

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
        result = np.array([closing(img, disk(5)) for img in remove_noise_2])
        return result

    def t2_tissues(self):
        median = np.array([nd.median_filter(img, footprint=disk(3)) for img in self.__t2]).astype(np.float64)
        thresh = median >= 0.1
        result = np.array([nd.median_filter(img, footprint=disk(3)) for img in thresh]).astype(np.float64)
        return result


def region_grow(vol, mask, start_point, epsilon=5, HU_mid=0, HU_range=0, fill_with=1):
    """
    `vol` your already segmented 3d-lungs, using one of the other scripts
    `mask` you can start with all 1s, and after this operation, it'll have 0's where you need to delete
    `start_point` a tuple of ints with (z, y, x) coordinates
    `epsilon` the maximum delta of conductivity between two voxels for selection
    `HU_mid` Hounsfield unit midpoint
    `HU_range` maximim distance from `HU_mid` that will be accepted for conductivity
    `fill_with` value to set in `mask` for the appropriate location in vol that needs to be flood filled
    """
    sizez = vol.shape[0] - 1
    sizex = vol.shape[1] - 1
    sizey = vol.shape[2] - 1

    items = []
    visited = []

    def enqueue(item):
        items.insert(0, item)

    def dequeue():
        s = items.pop()
        visited.append(s)
        return s

    enqueue((start_point[0], start_point[1], start_point[2]))

    while not items == []:

        z, x, y = dequeue()

        voxel = vol[z, x, y]
        mask[z, x, y] = fill_with

        if x < sizex:
            tvoxel = vol[z, x + 1, y]
            if abs(tvoxel - voxel) < epsilon and abs(tvoxel - HU_mid) < HU_range:
                enqueue((z, x + 1, y))

        if x > 0:
            tvoxel = vol[z, x - 1, y]
            if abs(tvoxel - voxel) < epsilon and abs(tvoxel - HU_mid) < HU_range:
                enqueue((z, x - 1, y))

        if y < sizey:
            tvoxel = vol[z, x, y + 1]
            if abs(tvoxel - voxel) < epsilon and abs(tvoxel - HU_mid) < HU_range:
                enqueue((z, x, y + 1))

        if y > 0:
            tvoxel = vol[z, x, y - 1]
            if abs(tvoxel - voxel) < epsilon and abs(tvoxel - HU_mid) < HU_range:
                enqueue((z, x, y - 1))

        if z < sizez:
            tvoxel = vol[z + 1, x, y]
            if abs(tvoxel - voxel) < epsilon and abs(tvoxel - HU_mid) < HU_range:
                enqueue((z + 1, x, y))

        if z > 0:
            tvoxel = vol[z - 1, x, y]
            if abs(tvoxel - voxel) < epsilon and abs(tvoxel - HU_mid) < HU_range:
                enqueue((z - 1, x, y))
