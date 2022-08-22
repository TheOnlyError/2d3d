import os
import time

from scipy.spatial.distance import cdist
# import tensorflow as tf
# from training.utils import ind2rgb, rgb2ind
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
from functools import partial
from multiprocessing import Pool

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def write(file, img):
    cv2.imwrite(file, img)


def main(config):
    path = 'annotations/hdd/' + config['dataset'] + '/'
    files = os.listdir(path)
    print(config)
    print(len(files))

    # done_lines = open('hm_cubi.log').readlines()[2:]
    # done = []
    # for l in done_lines:
    #     if l[0] == 'D':
    #         done.append(l.split('  ')[1].split(']')[0] + ']')
    #
    # files = [f for f in files if f not in done]
    # print(len(done))
    # print(len(files))

    # pool = Pool(processes=55)
    pool = Pool(processes=22)
    pool.map(partial(process, config, False), files)
    pool.close()
    pool.join()

    # for f in ['31878855.jpg_True_True_[]']:
    #     process(config, False, f)

    # step = 200
    # i = 0
    # start = i*step
    # end = start+step
    # files = files[start:end]
    # for f in files:
    #     process(config, False, f)


def generate_heatmap(c, show, mask):
    img = (mask == c + 1).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)

    if show:
        plt.figure(dpi=200)
        plt.imshow(mask)
        # plt.scatter(*zip(*centroids), color='b', s=.5)

    contours_map = np.zeros(img.shape, np.float32)

    endpoints = []
    for stat in stats[1:]:

        x = stat[0] - 1
        y = stat[1] - 1
        w, h = stat[2] + 1, stat[3] + 1
        if show:
            plt.gca().add_patch(Rectangle((x, y), w, h, facecolor='g', alpha=0.6))

        bb = img[y: y + h, x: x + w]
        contours, hierarchy = cv2.findContours(bb, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE, offset=[x, y])

        cv2.drawContours(contours_map, contours, -1, (255, 255, 255), -1)

        xy = []
        for contour in contours:
            for coord in contour:
                p = (coord[0][0], coord[0][1])
                if img[p[1], p[0]] == 1:
                    xy.append(p)

        x_min = (img.shape[1] + 1, 0)
        x_max = (-1, 0)
        y_min = (0, img.shape[0] + 1)
        y_max = (0, -1)
        p_min = (img.shape[0] + 1, img.shape[1] + 1)
        p_max = (-1, -1)
        xy = sorted(xy, key=lambda p: sum(p))
        for p in xy:
            if p[0] < x_min[0]:
                x_min = p
            elif p[0] > x_max[0]:
                x_max = p
            elif p[1] < y_min[1]:
                y_min = p
            elif p[1] > y_max[1]:
                y_max = p
            elif p[0] + p[1] < sum(p_min):
                p_min = p
            elif p[0] + p[1] > sum(p_max):
                p_max = p

        endpoints += xy
        # endpoints += list({x_min, x_max, y_min, y_max, p_min, p_max})

    if show:
        plt.axis('off')
        plt.savefig('heatmap_step1', bbox_inches='tight', pad_inches=0)

        plt.scatter(*zip(*endpoints), color='r', s=.1)
        plt.savefig('heatmap_step2', bbox_inches='tight', pad_inches=0)
        plt.show()

    heatmap = np.zeros(img.shape, np.float32)
    endpoints = endpoints[:-1]
    if len(np.array(endpoints).shape) == 2:
        for y, x in np.ndindex(heatmap.shape):
            heatmap[y, x] = np.min(cdist(np.array([(x, y)]), endpoints, 'sqeuclidean'))

    return heatmap, endpoints, contours_map


def process(config, show, f):
    tic = time.time()
    path = 'annotations/hdd/' + config['dataset'] + '/'
    input = cv2.imread(path + f + '/input.png')
    mask = cv2.imread(path + f + '/mask.png', 0)

    if show:
        plt.figure(dpi=200)
        plt.imshow(ind2rgb(mask))
        plt.axis('off')
        plt.savefig('heatmap_label', bbox_inches='tight', pad_inches=0)

    for c in config['classes_indices']:
        heatmap_path = path + f + '/heatmap_' + config['classes'][c] + '.npy'
        # if True or not os.path.isfile(heatmap_path):
        if not os.path.isfile(heatmap_path):
            data = generate_heatmap(c, show, mask)
            heatmap, endpoints, contours_map = data
            if len(np.array(endpoints).shape) != 2:
                data = [[], endpoints, []]
            np.save(path + f + '/heatmap_' + config['classes'][c], data)
        else:
            data = np.load(heatmap_path, allow_pickle=True)

        create_heatmap = True
        if create_heatmap:
            img = (mask == c + 1).astype(np.uint8)
            heatmap, endpoints, contours_map = data

            avg_heatmap = np.zeros(img.shape, np.float32)
            if len(np.array(endpoints).shape) != 2:
                print('EMPTY: {0}'.format(f))
            else:
                sigmas = [2, 10]
                for sigma in sigmas:
                    sigma = sigma ** 2
                    heatmap_beta = np.copy(heatmap)
                    for y, x in np.ndindex(heatmap_beta.shape):
                        heatmap_beta[y, x] = np.exp(-heatmap[y, x] / sigma)

                    heatmap_beta += contours_map / 255.0
                    heatmap_beta[heatmap_beta < 1e-7] = 0.0
                    heatmap_beta[heatmap_beta > 1] = 1.0

                    avg_heatmap += heatmap_beta
                    if show:
                        plt.figure(dpi=200)
                        plt.imshow(heatmap_beta)
                        plt.show()
                        print(np.min(heatmap_beta), np.max(heatmap_beta))

                        plt.figure(dpi=200)
                        plt.imshow(input)
                        plt.imshow(heatmap_beta, alpha=0.6)
                        plt.axis('off')
                        plt.savefig('heatmap_beta' + str(sigma), bbox_inches='tight', pad_inches=0)
                        plt.show()

                avg_heatmap /= len(sigmas)

            write(path + f + '/heatmap_' + config['classes'][c] + '.png', avg_heatmap * 255)

            if show:
                # plt.hist(heatmap.flatten())
                # plt.show()

                plt.figure(dpi=200)
                plt.imshow(mask)
                plt.axis('off')
                plt.show()

                print(np.min(heatmap), np.max(heatmap))
                print(np.min(avg_heatmap), np.max(avg_heatmap))

                plt.figure(dpi=200)
                # plt.imshow(input)
                # plt.imshow(avg_heatmap, alpha=0.6)
                plt.imshow(avg_heatmap)
                plt.axis('off')
                plt.savefig('avg_heatmap_betas', bbox_inches='tight', pad_inches=0)
                plt.show()

    toc = time.time()
    print('Done: ', f, (toc - tic) / 60)


def single(config):
    path = 'annotations/hdd/' + config['dataset'] + '/'
    files = os.listdir(path)
    process(config, False, files[0])


def plot_exp():
    x = np.linspace(0, 500, 2000)
    y1 = (np.exp(-x / 2 ** 2) + np.exp(-x / 10 ** 2)) / 2
    y1[y1 < 0.5] = np.nan
    y2 = (np.exp(-x / 2 ** 2) + np.exp(-x / 10 ** 2)) / 2
    y2[y2 >= 0.5] = np.nan
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, [0.5 for _ in x], linestyle='dashed')

    plt.xlabel('Euclidian distance (pixels)')
    plt.ylabel('Heatmap value')
    plt.legend([r'$\beta = 2$', r'$\beta = 10$'], loc='upper right')
    plt.savefig('heatmap_exp.png', pad_inches=0)
    plt.show()


def plot_scales():
    heatmap = cv2.imread('avg_heatmap_betas.png', 0)
    scales = [4, 8, 16]
    for s in scales:
        plt.figure(dpi=200)
        dims = [heatmap.shape[1] / s, heatmap.shape[0] / s]
        dims = [int(x) for x in dims]
        plt.imshow(cv2.resize(heatmap, dims, interpolation=cv2.INTER_AREA))
        plt.axis('off')
        plt.savefig('heatmap_scale{0}'.format(s) + '.png', bbox_inches='tight', pad_inches=0)
        plt.show()


def eval():
    prediction = cv2.cvtColor(cv2.imread('heatmap_label.png'), cv2.COLOR_BGR2RGB)
    prediction = np.expand_dims(rgb2ind(prediction), axis=0)
    y_pred = tf.one_hot(prediction, 3)[:, :, :, 2]

    y_pred = np.random.random(y_pred.shape)

    y_true = np.expand_dims(cv2.imread('avg_heatmap_betas.png', 0) / 255.0, axis=0)

    diff = np.square(y_pred - y_true)
    diff = np.linalg.norm(y_pred - y_true)
    print(diff)
    # loss = np.sum(diff)
    # print(1 - 1/loss)
    # plt.imshow(diff.squeeze())
    # plt.show()


if __name__ == '__main__':
    # eval()
    # exit(0)

    # plot_exp()
    # # plot_scales()
    # exit(0)

    configs = [
        # {
        #     'dataset': 'multi_plans_augment',
        #     'classes': ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all'],
        #     'classes_indices': [3, 4, 5, 6],  # TODO maybe add stairs
        # },
        {
            'dataset': 'cubicasa5k_augment',
            'classes': ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all'],
            'classes_indices': [3, 5, 6],
        },
        # {
        #     'dataset': 'r3d_augment',
        #     'classes': ['walls', 'openings'],
        #     'classes_indices': [1],
        # },
    ]

    # single(configs[0])
    # exit(0)

    for config in configs:
        tic = time.time()
        main(config)
        toc = time.time()
        print('total evaluation time = {} minutes'.format((toc - tic) / 60))
