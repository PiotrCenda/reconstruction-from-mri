import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from tqdm import tqdm

from data_rigid_transform import rigid_transform
from data_manipulation import func_timer


@func_timer
def plot_3d(image):
    image = image[1:-1, 1:-1, 1:-1]
    xm, ym, zm = np.mgrid[0:image.shape[0], 0:image.shape[2], 0:image.shape[1]].astype(np.float32)
    xm = xm * 5

    points = list()

    print(f"Creating points of 3d {image.__name__}: ")

    for d, x, y, z in tqdm(zip(image.ravel(), xm.ravel(), ym.ravel(), zm.ravel())):
        if d:
            points.append(np.array([x, y, z]))

    mesh = pv.PolyData(points)
    mesh.plot(point_size=7, style='points', color='white', eye_dome_lighting=True, render_points_as_spheres=True)


def plot_rigid_transform(img, rigid_params, slice_num=50, thresh=0.05):
    f, ax = plt.subplots(2, 2)
    plt.set_cmap('gray')

    ax[0][0].imshow(np.np.logical_not(img.t1[slice_num] > thresh))
    ax[0][0].set_title('t1')

    ax[0][1].imshow(img.t2[slice_num] > thresh)
    ax[0][1].set_title('t2')

    trans_t2 = rigid_transform(img.t2, rigid_params)

    ax[1][0].imshow(trans_t2[slice_num])
    ax[1][0].set_title('t2 transformed')

    ax[1][1].imshow(np.logical_xor(trans_t2 > thresh, np.np.logical_not(img.t1[slice_num] > 0.05))[slice_num])
    ax[1][1].set_title('trans t2-t1')

    plt.show()
