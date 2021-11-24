import numpy as np
import pyvista as pv
from skimage.measure import marching_cubes

from data_manipulation import func_timer


@func_timer
def plot_3d(image):
    points = point_cloud_from_mask(image)

    print("\nCreating model...")

    mesh = pv.PolyData(points)
    mesh.plot(point_size=7, style='points', color='white', eye_dome_lighting=True, cpos="yx",
              render_points_as_spheres=True)


@func_timer
def plot_3d_surface(image):
    print("\nCreating model surface...")
    points, faces, _, _ = marching_cubes(image, spacing=(5, 1, 1), allow_degenerate=True)

    faces = np.hstack([np.concatenate(([3], row)) for row in faces])

    mesh = pv.PolyData(points, faces)
    mesh.plot(eye_dome_lighting=True, cpos="yx")


def point_cloud_from_mask(img):
    image = np.array(img).astype(np.uint8)

    xm, ym, zm = np.mgrid[0:image.shape[0], 0:image.shape[2], 0:image.shape[1]].astype(np.float64)
    xm = xm * 5

    print("\nCreating points for 3d visualization...")
    points = list()

    for d, x, y, z in zip(image.ravel(), xm.ravel(), ym.ravel(), zm.ravel()):
        if d:
            points.append(np.array([x, y, z]))

    return points
