import numpy as np
import pyvista as pv
from tqdm import tqdm
from skimage.measure import marching_cubes_lewiner

from data_manipulation import func_timer


@func_timer
def plot_3d_points(image):
    image = image[1:-1, 1:-1, 1:-1]
    xm, ym, zm = np.mgrid[0:image.shape[0], 0:image.shape[2], 0:image.shape[1]].astype(np.float64)
    xm = xm * 5

    print(f"\nCreating points of 3d image: ")
    points = list()

    for d, x, y, z in tqdm(zip(image.ravel(), xm.ravel(), ym.ravel(), zm.ravel())):
        if d:
            points.append(np.array([x, y, z]))

    mesh = pv.PolyData(points)
    mesh.plot(point_size=7, style='points', color='white', eye_dome_lighting=True, cpos="yx",
              render_points_as_spheres=True)


@func_timer
def plot_3d_surface(image):
    image = image[1:-1, 1:-1, 1:-1]
    vertices, faces, _, _ = marching_cubes_lewiner(image, spacing=(5, 1, 1))

    faces = np.hstack([np.concatenate(([3], row)) for row in faces])

    mesh = pv.PolyData(vertices, faces)
    mesh_smoothed = mesh.smooth(n_iter=50)
    mesh_smoothed.plot(eye_dome_lighting=True, cpos="yx")
