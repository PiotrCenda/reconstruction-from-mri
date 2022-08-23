import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_loader import normalize
 

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(r, phi):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return(x, y)


def rotate(img):
    img = np.rot90(img, k=1, axes=(1, 2))
    img = np.rot90(img, k=1, axes=(0, 2))
    return img


def get_linspace(img, angle, z, points_nb=1000):
    angle_for_r = angle - 90 * np.floor(angle / 90)
    
    if round(angle_for_r, 6) >= 45:
        angle_for_r = 90 - round(angle_for_r, 6) 
    
    x_start = img.shape[0] // 2
    y_start = img.shape[1] // 2
    
    r = (img.shape[0] // 2) / np.cos(np.radians(angle_for_r))
    x_end, y_end = pol2cart(r, np.radians(angle))
    x_end +=  x_start
    y_end += y_start
    
    # print(f"r = {r}")
    # print(f"angle = {angle}")
    # print(f"angle_radians = {np.radians(angle)}")
    # print(f"cos = {np.cos(np.radians(angle))}")
    # print(f"x_end = {x_end}")
    # print(f"y_end = {y_end}")
        
    linepoints = [[x, y, z] for x, y in zip(np.linspace(x_start, x_end, points_nb), np.linspace(y_start, y_end, points_nb))]
    
    return(linepoints)


def get_pixel_from_points(img, angle, z):
    linepoints = get_linspace(img, angle, z)
    points = [img[round(x)-1, round(y)-1, round(z)-1] for x, y, z in linepoints]

    return np.mean(points)
    

def generate_pantomography(img, angle_mulitplication=5):
    angle_nb = 180*angle_mulitplication
    pantomography = np.zeros((angle_nb, img.shape[2] // 2))
    
    for z in tqdm(range(0, img.shape[2] // 2), desc="Calculating pantomogrphy"):
        for ang_nb in range(0, angle_nb):
            # print(f"\nGenerating pantomography for z = {z}, angle = {ang_nb/angle_mulitplication}")
            pantomography[ang_nb, z] = get_pixel_from_points(img, ang_nb/angle_mulitplication, z)
            
    pantomography = normalize(pantomography)
    
    print(pantomography.shape)
    print(pantomography)
    
    return pantomography
