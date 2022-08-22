import cv2
from matplotlib.patches import Rectangle
import numpy as np
from datasets.augment import rotate_image
from training.post_process import post_process
from training.utils import rgb2ind, ind2rgb, ind2rgba
import matplotlib.pyplot as plt

def write(file, img):
    cv2.imwrite(file + '.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # file = 'result20220806-1159020'
    # file = 'result_5_20220816-075226'
    # file = 'result_8_20220816-075233'
    # file = 'result20220802-1209040'
    # file = 'result20220802-1137100'
    # file = 'result20220728-1831220'

    file = 'result20220822-0301550'
    image = cv2.imread('results/' + file + '.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # image = rotate_image(image, 22, (255, 255, 255))
    # image = rotate_image(image, 45, (255, 255, 255))

    # crop = 70
    # image = image[crop:-crop, crop:-crop]

    dpi = 400

    plt.figure(dpi=dpi)
    plt.imshow(image)
    # rect = Rectangle((115, 190), 40, 40, linewidth=1, edgecolor='r', facecolor='none')
    # plt.gca().add_patch(rect)
    plt.axis('off')
    # plt.savefig('results/zeng_pp1', bbox_inches='tight', pad_inches=0)
    # plt.savefig('results/pp_original', bbox_inches='tight', pad_inches=0)
    write('results/pp_original', image)
    plt.show()

    results = post_process(rgb2ind(image), eval=True, high_res=False, debug=True)
    resized_results = []
    for result_pp in results[1:]:
        print(np.unique(result_pp))
        result_pp = ind2rgb(result_pp)
        # result_pp = cv2.resize(result_pp, [image.shape[1], image.shape[0]], interpolation=cv2.INTER_NEAREST)
        result_pp = cv2.resize(result_pp, [image.shape[1], image.shape[0]], interpolation=cv2.INTER_AREA)
        plt.figure(dpi=dpi)
        plt.imshow(result_pp)
        resized_results.append(result_pp)

        # rect = Rectangle((115, 190), 40, 40, linewidth=1, edgecolor='r', facecolor='none')
        # plt.gca().add_patch(rect)

        plt.axis('off')
        plt.show()

    # write('results/pp_result_approximate_' + str(scale), resized_results[0])
    write('results/pp_result_refined', resized_results[0])
    # write('results/pp_result_heuristic_' + str(scale), resized_results[1])
