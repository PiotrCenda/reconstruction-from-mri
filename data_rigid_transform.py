import numpy as np
import scipy.ndimage as nd
from scipy import optimize


# Modeling images
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


def translate_matrix(x, y, z):
    """
    returns rotation matrix for axis
    theta should be in radians
    """
    trans_mat = np.array([[1, 0, 0, x],
                          [0, 1, 0, y],
                          [0, 0, 1, z],
                          [0, 0, 0, 1]])

    return trans_mat


def scale_matrix(x, y, z):
    """
    returns scale matrix for axis
    theta should be in radians
    """
    scale_mat = np.array([[x, 0, 0, 0],
                          [0, y, 0, 0],
                          [0, 0, z, 0],
                          [0, 0, 0, 1]])

    return scale_mat


def center_matrix(transform, shape):
    x_mid = int((shape[1] - 1) / 2)
    y_mid = int((shape[0] - 1) / 2)
    z_mid = int((shape[2] - 1) / 2)
    # A @ transform @ A^(-1)
    a = np.array([[1, 0, 0, x_mid],
                  [0, 1, 0, y_mid],
                  [0, 0, 1, z_mid],
                  [0, 0, 0, 1]]).reshape(4, 4)

    return a @ transform @ np.linalg.pinv(a)


def axises_rotations_matrix(theta1, theta2, theta3):
    return rotation_matrix_x(theta1) @ rotation_matrix_y(theta2) @ rotation_matrix_z(theta3)


def rigid_transform(img, args):
    alpha, beta, gamma, x, y, z, = args[0], args[1], args[2], args[3], args[4], args[5]

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
    transform_rotation_matrix = axises_rotations_matrix(alpha, beta, gamma) @ translate_matrix(x, y, z)
    centered_transform_rotation_matrix = center_matrix(transform_rotation_matrix, img.shape)

    # calculate new coordinates
    m_x_transformed = np.linalg.pinv(centered_transform_rotation_matrix) @ m_x

    trans_grid_x = m_x_transformed[1].reshape(grid_x.shape)
    trans_grid_y = m_x_transformed[0].reshape(grid_y.shape)
    trans_grid_z = m_x_transformed[2].reshape(grid_z.shape)
    trans_grids = np.array([trans_grid_x, trans_grid_y, trans_grid_z])

    grid = trans_grids

    transformed_image = nd.map_coordinates(img, grid, order=0)

    return transformed_image


def ssd(a, b):
    dif = a.ravel() - b.ravel()
    return np.dot(dif, dif)


def register_image(image_model, image_to_change):
    start_params = np.array([0, 0, 0, 0, 0, 0])

    def cost_function(params):
        image_changed = rigid_transform(image_to_change, params)
        print(params)
        return ssd(image_changed, image_model)

    best_parameters = optimize.fmin(func=cost_function, x0=start_params)
    return best_parameters
