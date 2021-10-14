import matplotlib.pyplot as plt
import os

from data_loader import read_data


def prepare_data():
    images =  read_data()
    return images

def create_list_of_masks(img, thresh_precision = 10):
    OUTPUT_FOLDER = 'test_masks'
    new_folder = str(len(os.listdir(OUTPUT_FOLDER)))
    os.mkdir(os.path.join(OUTPUT_FOLDER, new_folder))
    thresh_list = [(1/(thresh_precision))*(i+1) for i in range(thresh_precision)]
    thresh_list_len = len(thresh_list)

    for i in range(thresh_list_len):
        for j in range(i+1, thresh_list_len):
            title = 'down_val_' + str(i) + '_up_val_' + str(j) + '.png'
            path = os.path.join(OUTPUT_FOLDER, new_folder, title)
            print(path)

            temp_th_img_t1 = img.thresh('T1',thresh_list[i], thresh_list[j])
            temp_th_img_t2 = img.thresh('T2',thresh_list[i], thresh_list[j])
            temp_mask_img_t1 = img.mask('T1',thresh_list[i], thresh_list[j])
            temp_mask_img_t2 = img.mask('T2',thresh_list[i], thresh_list[j])

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
            f.suptitle('Lower value: ' + str(thresh_list[i]) + ' Upper Value: ' + str(thresh_list[j]))
            f.savefig(path)
            plt.close(f)

            del(temp_th_img_t1)
            del(temp_th_img_t2)
            del(temp_mask_img_t1)
            del(temp_mask_img_t2)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img = prepare_data()
    create_list_of_masks(img, 20)

