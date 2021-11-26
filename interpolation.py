import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def scale_z_to_y(img):
    aspect_ratio = img.shape[1]/img.shape[0]
    my_dpi = int(img.shape[1]/8)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 8)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    for slice in range(img.shape[2]):
        print(slice)
        ax.imshow(img[:, slice, :], interpolation='quadric', cmap='gray', aspect=aspect_ratio)
        fig.savefig('temp/' + str(slice) + '.png', dpi=my_dpi)

def zy_to_tif():
    img = []
    directory = 'temp'
    img_paths = [os.path.join(directory, f) for f in os.listdir(directory)]
    img_paths.sort(key=len)
    for path in img_paths:
        # print(path)
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img.append(im)
    img = np.array(img)
    img = np.swapaxes(img, 0, 1)
    return img

def show_xyz(img):
    z_slice = img.shape[0] // 2
    y_slice = img.shape[1] // 2
    x_slice = img.shape[2] // 2

    f, ax = plt.subplots(1,3)
    ax[0].imshow(img[z_slice, :, :], cmap = 'gray')
    ax[1].imshow(img[::-1, y_slice, :], cmap = 'gray')
    ax[2].imshow(img[::-1, :, x_slice], cmap = 'gray')
    plt.show()

def show_rec_xyz(img):
    z_slice = np.zeros(img[0, :, :].shape)
    y_slice = np.zeros(img[:, 0, :].shape)
    x_slice = np.zeros(img[:, :, 0].shape)
    for i in range(img.shape[0]):
        z_slice += img[i, :, :]
    for i in range(img.shape[1]):
        y_slice += img[::-1, i, :]
    for i in range(img.shape[2]):
        x_slice += img[::-1, :, i]

    f, ax = plt.subplots(1,3)
    ax[0].imshow(z_slice, cmap = 'gray')
    ax[1].imshow(y_slice, cmap = 'gray')
    ax[2].imshow(x_slice, cmap = 'gray')
    plt.show()

def slices_to_array(img):
    pass

