import numpy as np
import scipy.ndimage as nd
from scipy import optimize
from skimage.segmentation import flood
from skimage.morphology import remove_small_holes, remove_small_objects, closing, disk
from skimage.feature import canny

from data_manipulation import timer_block


def rotation_matrix_x(theta):
    """
    returns rotation matrix for axis,
    theta should be in radians
    """
    rot_mat = np.array([[1, 0, 0, 0],
                        [0, np.cos(theta), np.sin(theta), 0],
                        [0, -np.sin(theta), np.cos(theta), 0],
                        [0, 0, 0, 1]])

    return rot_mat


def rotation_matrix_y(theta):
    """
    returns rotation matrix for axis,
    theta should be in radians
    """
    rot_mat = np.array([[np.cos(theta), 0, -np.sin(theta), 0],
                        [0, 1, 0, 0],
                        [np.sin(theta), 0, np.cos(theta), 0],
                        [0, 0, 0, 1]])

    return rot_mat


def rotation_matrix_z(theta):
    """
    returns rotation matrix for axis
    theta should be in radians
    """
    rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                        [np.sin(theta), np.cos(theta), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

    return rot_mat


def translate_matrix(x, y, z, hxy, hxz, hyz, hyx, hzx, hzy, sx, sy, sz):
    """
    returns rotation matrix for axis
    theta should be in radians
    """
    trans_mat = np.array([[sx, hxy, hxz, x],
                          [hyx, sy, hyz, y],
                          [hzx, hzy, sz, z],
                          [0, 0, 0, 1]])
    return trans_mat


def center_matrix(transform, shape):
    x_mid = int((shape[1] - 1) / 2)
    y_mid = int((shape[0] - 1) / 2)
    z_mid = int((shape[2] - 1) / 2)

    a = np.array([[1, 0, 0, x_mid],
                  [0, 1, 0, y_mid],
                  [0, 0, 1, z_mid],
                  [0, 0, 0, 1]]).reshape(4, 4)

    return a @ transform @ np.linalg.pinv(a)


def axises_rotations_matrix(theta1, theta2, theta3):
    return rotation_matrix_x(theta1) @ rotation_matrix_y(theta2) @ rotation_matrix_z(theta3)


def rigid_transform(img, args):
    alpha, beta, gamma, x, y, z, hxy, hxz, hyz, hyx, hzx, hzy, sx, sy, sz = args[0], args[1], args[2], args[3], args[4], args[5], \
                                                             args[6], args[8], args[7], args[9], args[10], args[11], args[12], args[13], args[14]

    # coordinates for 3d image
    grid_x, grid_y, grid_z = np.meshgrid(np.arange(img.shape[1]),
                                         np.arange(img.shape[0]),
                                         np.arange(img.shape[2]))
    new_grid_x = np.ndarray.flatten(grid_x)
    new_grid_y = np.ndarray.flatten(grid_y)
    new_grid_z = np.ndarray.flatten(grid_z)

    m_x = np.array([new_grid_x,
                    new_grid_y,
                    new_grid_z, np.ones(new_grid_x.shape)])

    # rotate matrix
    transform_rotation_matrix = axises_rotations_matrix(alpha, beta, gamma) \
                                @ translate_matrix(x, y, z, hxy, hxz, hyz, hyx, hzx, hzy, sx, sy, sz)
    centered_transform_rotation_matrix = center_matrix(transform_rotation_matrix, img.shape)

    # calculate new coordinates
    m_x_transformed = np.linalg.inv(centered_transform_rotation_matrix) @ m_x

    trans_grid_x = m_x_transformed[1].reshape(grid_x.shape)
    trans_grid_y = m_x_transformed[0].reshape(grid_y.shape)
    trans_grid_z = m_x_transformed[2].reshape(grid_z.shape)
    trans_grids = np.array([trans_grid_x, trans_grid_y, trans_grid_z])

    grid = trans_grids

    transformed_image = nd.map_coordinates(img, grid, order=0)

    return transformed_image


def model_to_register_fitting(image, flood_thresh=0.05):
    median = np.array([nd.median_filter(img, footprint=disk(2)) for img in image]).astype(np.float64)
    model = np.array([flood(img, (0, 0), tolerance=flood_thresh) for img in median])
    closed = np.array([closing(img, disk(5)) for img in model])
    remove_noise = np.array([remove_small_holes(img, area_threshold=1500) for img in closed])
    remove_noise2 = np.array([remove_small_objects(img, min_size=1500) for img in remove_noise])
    median2 = np.array([nd.median_filter(img, footprint=disk(2)) for img in remove_noise2])
    return np.array([canny(img, sigma=2) for img in median2]).astype(np.bool_)


def ssd(a, b):
    err = np.logical_xor(a[1:-1, 1:-1, 1:-1], b[1:-1, 1:-1, 1:-1]).astype(np.int64)
    cost = np.sqrt(np.sum([img.ravel() for img in err]))
    print(f"Cost function: {cost}")
    return cost


def register_image(image_model, image_to_change):
    # start_params_without_shearing -> np.array([0, 0, 0, 3.50713579e-04, 2.56382387e-04, -2.36681565e-04,
    #                                            9.40513164e-01, 9.38176829e-01, 1.00128937e+00])

    start_params = np.array([1.51709147e+00, 1.18339327e+00, -1.17477642e-02, 7.45291130e-02, -1.40245845e+00,
                             4.35379594e-01, -2.87405872e-01, -4.27344509e+01, 7.68620321e-12, -3.70788805e-03,
                             -7.19916508e-02, -2.79734267e-03])

    def cost_function(params):
        image_changed = rigid_transform(image_to_change, params)
        print(f"Checking parameters: {params}")
        return ssd(image_changed, image_model)

    best_parameters = optimize.fmin_powell(func=cost_function, x0=start_params)

    return best_parameters


def auto_t1_t2_fitting(img):
    t1 = model_to_register_fitting(img.t1, flood_thresh=0.05)
    t2 = model_to_register_fitting(img.t2, flood_thresh=0.03)

    with timer_block('t2 to t1 fitting'):
        params = register_image(image_model=t1, image_to_change=t2)

    print(f"\nFinal rigid parameters: {params}")

    img.t2_rigid_transform(parameters=params)
