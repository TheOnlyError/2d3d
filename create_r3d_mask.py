import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


def write(file, img):
    cv2.imwrite(file, img)


def main():
    path = 'annotations/hdd/r3d/'
    files = os.listdir(path)
    j = 0

    files = list(set([f[:-4].split('_')[0] for f in files]))

    for f in files:
        j += 1
        print(j, f[0])

        classes = ['wall', 'close']

        openings = cv2.imread(path + f + '_close.png', 0)

        mask = np.zeros(openings.shape, np.uint8)
        for c in range(len(classes)):
            img = cv2.imread(path + f + '_' + classes[c] + '.png', 0)
            if img is None:
                continue
            c_ind = (img > 0.5).astype(np.uint8)
            mask[c_ind == 1] = c + 1

        write(path + f + '_mask.png', mask)


if __name__ == '__main__':
    main()
