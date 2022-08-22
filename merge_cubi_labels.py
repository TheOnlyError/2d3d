import os
import numpy as np
import cv2

from parse_cubi_labels import parse_cubi_label
from training import utils
from training.utils import ind2rgb


def write(file, img):
    cv2.imwrite(file, img)

def parse_all():
    i = 'cubicasa5k_single'
    paths = []
    paths += open(i + '_train.txt', 'r').read().splitlines()
    paths += open(i + '_val.txt', 'r').read().splitlines()
    paths += open(i + '_test.txt', 'r').read().splitlines()

    files = []
    for p in paths:
        files.append(p.split('\t')[0].split('/')[1])
    for file in files:
        print(file)
        main(file)

def main(file):
    path = 'annotations/hdd/cubicasa5k_fix/' + file

    parse_cubi_label(path)

    classes = ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all']
    i = 0
    kernel = np.ones((3, 3), np.uint8)

    doors = cv2.imread(path + '/door.png', 0)
    # doors = cv2.dilate(doors, kernel)
    doors[doors > 0] = 255

    windows = cv2.imread(path + '/window.png', 0)
    windows = cv2.dilate(windows, kernel)
    windows[windows > 0] = 255

    columns = cv2.imread(path + '/column.png', 0)
    columns[columns > 0] = 255

    railings = cv2.imread(path + '/railing.png', 0)
    railings[railings > 0] = 255

    stairs = cv2.imread(path + '/stairs.png', 0).astype('int32')

    walls = cv2.imread(path + '/wall.png', 0).astype('int32')

    # if len({walls.shape, doors.shape, windows.shape, columns.shape, stairs.shape, railings.shape}) > 1:
    #     print('INCOMPATIBLE')
    #     exit(0)

    empty = np.zeros(walls.shape, np.uint8)

    # stairs = stairs - railings
    # stairs[stairs < 0] = 0
    # stairs[stairs > 0] = 255
    stairs[(railings > 0)] = 0

    walls = walls - doors - windows + columns
    walls[walls < 0] = 0
    walls[walls > 0] = 255
    write(path + '/' + classes[i] + '.png', walls)
    i += 1

    # No glass walls
    write(path + '/' + classes[i] + '.png', empty)
    i += 1

    write(path + '/' + classes[i] + '.png', railings)
    i += 1

    write(path + '/' + classes[i] + '.png', doors)
    i += 1

    # No sliding doors
    write(path + '/' + classes[i] + '.png', empty)
    i += 1

    write(path + '/' + classes[i] + '.png', windows)
    i += 1

    write(path + '/' + classes[i] + '.png', stairs)
    i += 1

    mask = np.zeros(doors.shape, np.uint8)
    for c in range(len(classes)):
        c_ind = (cv2.imread(path + '/' + classes[c] + '.png', 0) > 0.5).astype(np.uint8)
        mask[c_ind == 1] = c + 1

    write(path + '/mask.png', mask)

    overlay = False
    if overlay:
        import matplotlib.pyplot as plt
        plt.figure(dpi=400)
        image = cv2.imread(path + '/F1_original.png')
        h,w,_ = image.shape
        plt.imshow(cv2.imread(path + '/F1_original.png').astype(np.uint8))
        plt.imshow(cv2.resize(ind2rgb(mask), (w,h)), alpha=0.6)
        plt.savefig(path + '/overlay')
        # plt.show()
        plt.close()


if __name__ == '__main__':
    # file = '4566'
    # main(file)

    # parse_all()

    files = ['1105', '215']
    for file in files:
        main(file)

