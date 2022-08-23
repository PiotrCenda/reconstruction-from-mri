import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
 

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
    angle = angle - 90 * np.floor(angle / 90)
    
    if round(angle, 6) >= 45:
        angle = 90 - round(angle, 6) 
    
    angle_radians = np.radians(angle)
    x_start = img.shape[0] // 2
    y_start = img.shape[1] // 2
    
    r = img.shape[0] / np.cos(angle_radians)
    x_end, y_end = pol2cart(r, angle_radians)
    
    # print(r)
    # print(angle)
    # print(np.cos(angle_radians))
    # print(x_end)
    # print(y_end)
        
    linepoints = [[x, y, z] for x, y in zip(np.linspace(x_start, x_end, points_nb), np.linspace(y_start, y_end, points_nb))]
    
    return(linepoints)


def get_pixel_from_points(img, angle, z):
    linepoints = get_linspace(img, angle, z)
    points = [img[round(x)-1, round(y)-1, round(z)-1] for x, y, z in linepoints]

    return np.mean(points)
    

def generate_pantomography(img, angle_mulitplication=4):
    angle_nb = 180*angle_mulitplication
    pantomography = np.zeros((angle_nb, img.shape[2]))
    
    for z in tqdm(range(0, img.shape[2] // 2), desc="Calculating pantomogrphy"):
        for ang_nb in range(0, angle_nb):
            # print(f"Generating pantomography for z = {z}, angle = {ang_nb/angle_mulitplication}")
            pantomography[ang_nb, z] = get_pixel_from_points(img, ang_nb/angle_mulitplication, z)
            
    print(pantomography.shape)
    print(pantomography)
    plt.imshow(pantomography, cmap='gray')
    plt.show()
    
    return pantomography
