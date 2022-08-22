import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from training.utils import ind2rgba, ind2rgb


def write(file, img):
    cv2.imwrite(file, img)


def main():
    files = os.walk('annotations/hdd/multi_plans')
    j = 0

    for f in files:
        j += 1
        if 'walls.png' not in f[2]:
            continue
        print(j, f[0])

        if 'mask.png' in f[2]:
            path = f[0].replace('\\', '/')
            img = cv2.imread(path + '/image.jpg')
            width = 3000
            height = int(width / img.shape[1] * img.shape[0])
            image = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(cv2.imread(path + '/mask.png', 0), (width, height), interpolation=cv2.INTER_NEAREST)
            plt.figure(dpi=800)
            plt.imshow(image)
            plt.imshow(ind2rgb(mask), alpha=0.75)
            # plt.imshow(ind2rgba(mask))
            plt.axis('off')
            plt.savefig('mufpo' + str(j) + '.png', bbox_inches='tight', pad_inches=0)
            plt.show()
            continue

        path = f[0].replace('\\', '/')

        classes = ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs']

        doors = cv2.imread(path + '/doors.png', 0)

        mask = np.zeros(doors.shape, np.uint8)
        for c in range(len(classes)):
            img = cv2.imread(path + '/' + classes[c] + '.png', 0)
            if img is None:
                continue
            c_ind = (img > 0.5).astype(np.uint8)
            mask[c_ind == 1] = c + 1


        write(path + '/mask.png', mask)


if __name__ == '__main__':
    main()
