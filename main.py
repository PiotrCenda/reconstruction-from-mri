import os
import numpy as np
from skimage import feature
from skimage.filters import median
from skimage.morphology import disk

from data_loader import read_data_from_folder
from data_manipulation import save_tif, timer_block, doce
from data_plotting import plot_3d
from data_rigid_transform import register_image, rigid_transform

params = np.array([0, 0, 0, 0, 0, 0, 0.95, 0.96, 1])

# TODO: make masks
# TODO: make automatic rigid transform (?)


if __name__ == '__main__':
    img = read_data_from_folder(os.path.abspath('data/head'))
    # img.t2_rigid_transform(parameters = params)

    model = np.array([median(img, selem=disk(3)) for img in (img.t1 > 0.05).astype(np.uint8)])
    img_to_change = np.array([median(img, selem=disk(3)) for img in (img.t2 > 0.05).astype(np.uint8)])

    save_tif(model, img_name="model")
    save_tif(img_to_change, img_name="img_to_change")

    with timer_block('t2 to t1 registration'):
        params = register_image(image_model=model, image_to_change=img_to_change)

    print(params)

    with timer_block('bones mask making'):
        bones = img.bones_mask()
        save_tif(bones, img_name="bones_mask")
        plot_3d(bones)
