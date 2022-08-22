import os
import numpy as np
import cv2

def write(file, img):
    cv2.imwrite(file, img)


def main():
    # files = os.walk('annotations/hdd/cubicasa5k')
    files = os.walk('annotations/hdd/cubicasa5k_single')
    j = 0

    for f in files:
        j += 1
        if 'model.svg' not in f[2] or 'stairs_all.png' not in f[2]: # or 'mask.png' in f[2]:
            continue
        print(j, f[0])
        if j == 876 or j == 1810:
            continue

        path = f[0].replace('\\', '/')

        # classes = ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all']
        classes = ['walls', 'railings', 'doors', 'windows', 'stairs_all']

        doors = cv2.imread(path + '/door.png', 0)

        mask = np.zeros(doors.shape, np.uint8)
        for c in range(len(classes)):
            c_ind = (cv2.imread(path + '/' + classes[c] + '.png', 0) > 0.5).astype(np.uint8)
            mask[c_ind == 1] = c + 1

        write(path + '/mask.png', mask)


if __name__ == '__main__':
    main()
