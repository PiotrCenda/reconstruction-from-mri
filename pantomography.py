import numpy as np
 

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(r, phi):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return(x, y)


def get_linspace(img, z, angle):
    angle_radians = np.radians(angle)
    x_start = img.shape[0] // 2
    y_start = img.shape[1] // 2
    r = img.shape[0] / np.cos(angle_radians)
    print(f"r = {r}, angle_rad = {angle_radians}")
    x_end, y_end = pol2cart(r, angle_radians)
    print(f"x_end = {x_end}, y_end = {y_end}")
    linepoints = [[x, y, z] for x, y in zip(np.linspace(x_start, x_end), np.linspace(y_start, y_end))]
    print(f"linepoints = {linepoints}")
    return(linepoints)


def rotate(img):
    img = np.rot90(img, k=1, axes=(1, 2))
    img = np.rot90(img, k=1, axes=(0, 2))
    return img


def generate_pantomography(img):
    pantomography = np.zeros((180*4, img.shape[2]))
    return pantomography
