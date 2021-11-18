import os
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from tqdm import tqdm


def plot_3d(image):
    xm, ym, zm = np.mgrid[0:image.shape[1]:5, 0:image.shape[2]:5, 0:image.shape[0]].astype(np.float32)

    points = list()

    for d, x, y, z in tqdm(zip(image[::5, ::5, :].ravel(), xm.ravel(), ym.ravel(), zm.ravel())):
        if d:
            points.append(np.array([x, y, z]))

    mesh = pv.PolyData(points)
    mesh.plot(point_size=2, style='points', color='red', smooth_shading=True)


def create_list_of_masks(img, thresh_precision=10):
    output_folder = os.path.join("results", 'test_masks')
    new_folder = str(len(os.listdir(output_folder)))
    os.mkdir(os.path.join(output_folder, new_folder))
    thresh_list = [(1 / thresh_precision) * (i + 1) for i in range(thresh_precision)]
    thresh_list_len = len(thresh_list)

    for i in range(thresh_list_len):
        for j in range(i + 1, thresh_list_len):
            title = 'down_val_' + str(i) + '_up_val_' + str(j) + '.png'
            path = os.path.join(output_folder, new_folder, title)
            print(path)

            temp_th_img_t1 = img.thresh('T1', thresh_list[i], thresh_list[j])
            temp_th_img_t2 = img.thresh('T2', thresh_list[i], thresh_list[j])
            temp_mask_img_t1 = img.mask('T1', thresh_list[i], thresh_list[j])
            temp_mask_img_t2 = img.mask('T2', thresh_list[i], thresh_list[j])

            f, ax = plt.subplots(2, 2)
            ax[0][0].imshow(temp_th_img_t1.t1[img.middle], cmap='gray')
            ax[0][0].set_title('T1 Tresh')
            ax[0][0].axis('off')
            ax[0][1].imshow(temp_th_img_t2.t2[img.middle], cmap='gray')
            ax[0][1].set_title('T2 Tresh')
            ax[0][1].axis('off')
            ax[1][0].imshow(temp_mask_img_t1.t1[img.middle], cmap='gray')
            ax[1][0].set_title('T1 Mask')
            ax[1][0].axis('off')
            ax[1][1].imshow(temp_mask_img_t2.t2[img.middle], cmap='gray')
            ax[1][1].set_title('T2 Mask')
            ax[1][1].axis('off')
            # plt.show()
            f.suptitle('Lower value: ' + str(round(thresh_list[i], 3)) +
                       ' Upper Value: ' + str(round(thresh_list[j], 3)))
            f.savefig(path)
            plt.close(f)

            del temp_th_img_t1
            del temp_th_img_t2
            del temp_mask_img_t1
            del temp_mask_img_t2
