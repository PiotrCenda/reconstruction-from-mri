class ImageSequences:
    """
    Class where T1 and T2 sequences are stored. Additionally there are functions which thresholds, masks and segments
    objects on images.
    """

    def __init__(self, img_dict):
        self.__all = img_dict
        self.__t1 = img_dict['T1']
        self.__t2 = img_dict['T2']

    def __copy__(self, data_dict=None):
        copy = ImageSequences(self.__all)
        return copy

    @property
    def t1(self):
        return self.__t1

    @property
    def t2(self):
        return self.__t2

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

    def background_mask(self):
        print(self.t1.shape)
        print(self.t2.shape)
