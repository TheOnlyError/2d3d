import math
import random

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = None

colors = {
    'R3D': 'tab:orange',
    'CubiCasa': 'tab:blue',
    'MUFP': 'tab:green',
}


def dimensions(dataset):
    train_paths = open(dataset + '_train.txt', 'r').read().splitlines()
    val_paths = open(dataset + '_val.txt', 'r').read().splitlines()
    test_paths = open(dataset + '_test.txt', 'r').read().splitlines()
    paths = train_paths + val_paths + test_paths
    print(len(paths))

    minSize = 100000
    minImg = ''
    maxSize = 0
    maxImg = ''
    minDim = (100000, 100000)
    minDimImg = ''
    maxDim = (0, 0)
    maxDimImg = ''
    for path in paths:
        img = cv2.imread('annotations/hdd/' + path.split('\t')[0])

        minSize = min(min(img.shape[0], img.shape[1]), minSize)
        maxSize = max(max(img.shape[0], img.shape[1]), maxSize)

        if img.shape[0] * img.shape[1] < minDim[0] * minDim[1]:
            minDim = img.shape
            minDimImg = path.split('\t')[0].split('/')[1]
        elif img.shape[0] * img.shape[1] > maxDim[0] * maxDim[1]:
            maxDim = img.shape
            maxDimImg = path.split('\t')[0].split('/')[1]

    print(minSize, minImg)
    print(maxSize, maxImg)
    print(minDim, minDimImg)
    print(maxDim, maxDimImg)
    print("Done")


def size(dataset):
    train_paths = open(dataset + '_train.txt', 'r').read().splitlines()
    val_paths = open(dataset + '_val.txt', 'r').read().splitlines()
    test_paths = open(dataset + '_test.txt', 'r').read().splitlines()
    paths = train_paths + val_paths + test_paths
    print(len(paths))


def get_statistics(dataset, classes):
    train_paths = open(dataset + '_train.txt', 'r').read().splitlines()
    val_paths = open(dataset + '_val.txt', 'r').read().splitlines()
    test_paths = open(dataset + '_test.txt', 'r').read().splitlines()
    paths = train_paths + val_paths + test_paths

    # paths = paths[:int(len(paths)/7)]
    # paths = paths[:int(len(paths)/5)]
    print(len(paths))
    xs = []
    ys = []
    weights = [0 for _ in classes]
    frequencies = [[] for _ in classes]
    for path in paths:
        img = cv2.imread('annotations/hdd/' + path.split('\t')[0])
        xs.append(img.shape[0])
        ys.append(img.shape[1])

        mask = cv2.imread('annotations/hdd/' + path.split('\t')[1], 0).astype(np.uint8)
        for c in range(len(classes)):
            weights[c] += np.count_nonzero(mask == c)
            frequencies[c].append(np.count_nonzero(mask == c) / (mask.shape[0] * mask.shape[1]))

    return xs, ys, weights, frequencies


def bin_dims(data, labels, type):
    # plt.figure(dpi=400)
    # plt.hist([[w * h for (w, h) in zip(d[0], d[1])] for d in data], bins=20)
    # plt.legend(labels, loc="upper right")
    # plt.yscale('log')
    # plt.show()

    plt.figure(dpi=300)
    for i, d in enumerate(data):
        # plt.scatter(d[0], d[1], alpha=0.5, lw=0, c=colors[labels[i]])
        plt.scatter(d[0], d[1], alpha=0.333, lw=0, c=colors[labels[i]])
    plt.xlabel('Height (pixels)')
    plt.ylabel('Width (pixels)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig('bin_dims_' + type, bbox_inches='tight')
    plt.show()
    exit(0)


def plot_areas(data, ticks=None, labels=None, type=''):
    if ticks is not None:
        ticks = [t[0].upper() + t[1:] for t in ticks]
    x = np.arange(len(data[0]))
    width = 0.2
    half_len = int(len(data) / 2)
    step = 1 if (2 * half_len < len(data)) else 0.5
    for i, val in enumerate(data[:half_len]):
        plt.bar(x - (i + step) * width, val, width, color=[colors[l] for l in labels[:half_len]])
    if step == 1:
        plt.bar(x, data[half_len], width, color=colors[labels[half_len]])
    if len(data) > 1:
        for i, val in enumerate(data[-half_len:]):
            plt.bar(x + (i + step) * width, val, width, color=[colors[l] for l in labels[-half_len:]])

    plt.xticks(x, ticks)
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.yscale('log')
    plt.ylabel('Number of pixels')
    plt.legend(labels, loc="upper right")
    plt.savefig('plot_areas_' + type, bbox_inches='tight')
    plt.show()


def plot_frequencies(data, ticks=None, labels=None, type=''):
    if ticks is None:
        ticks = []
    for t in range(len(ticks)):
        points = []
        weights = []
        for d in data:
            points.append(d[t])
            weights.append(np.ones_like(d[t]) / len(d[t]))
        # plt.hist(points, weights=weights, color=[colors[l] for l in labels], bins=8)
        plt.hist(points, weights=weights, color=[colors[l] for l in labels], bins=12)
        plt.xlabel('Ratio of pixels')
        plt.ylabel('Normalized frequency')
        plt.legend(labels, loc="upper right")
        plt.savefig('plot_frequencies_{0}_{1}'.format(type, ticks[t]), bbox_inches='tight')
        plt.show()


def map_weights(weights, mapping, ticks):
    weights_map = [0 for _ in ticks]
    for i in range(len(weights)):
        if mapping[i] is not None:
            weights_map[mapping[i]] += weights[i]
    return weights_map


def map_frequencies(weights, mapping, ticks):
    weights_map = [[] for _ in ticks]
    for i in range(len(weights)):
        if mapping[i] is not None:
            weights_map[mapping[i]].append(weights[i])

    return [sum(w) / len(w) for w in weights_map]


if __name__ == '__main__':
    r3d_classes = ['background', 'walls', 'openings']
    multi_classes = ['background', 'walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs']

    # dimensions('r3d')
    # dimensions('cubicasa5k')
    # dimensions('multi_plans')
    # exit(0)

    # size('cubicasa5k_augment')

    r3d_xs, r3d_ys, r3d_weights, r3d_frequencies = get_statistics('r3d', r3d_classes)
    np.save('r3d_xs', r3d_xs)
    np.save('r3d_ys', r3d_ys)
    np.save('r3d_weights', r3d_weights)
    np.save('r3d_frequencies', r3d_frequencies)

    cubi_xs, cubi_ys, cubi_weights, cubi_frequencies = get_statistics('cubicasa5k', multi_classes)
    np.save('cubi_xs', cubi_xs)
    np.save('cubi_ys', cubi_ys)
    np.save('cubi_weights', cubi_weights)
    np.save('cubi_frequencies', cubi_frequencies)

    multi_xs, multi_ys, multi_weights, multi_frequencies = get_statistics('multi_plans', multi_classes)
    np.save('multi_xs', multi_xs)
    np.save('multi_ys', multi_ys)
    np.save('multi_weights', multi_weights)
    np.save('multi_frequencies', multi_frequencies)

    r3d_xs = np.load('r3d_xs.npy')
    r3d_ys = np.load('r3d_ys.npy')
    r3d_weights = np.load('r3d_weights.npy')
    r3d_frequencies = np.load('r3d_frequencies.npy')
    cubi_xs = np.load('cubi_xs.npy')
    cubi_ys = np.load('cubi_ys.npy')
    cubi_weights = np.load('cubi_weights.npy')
    cubi_frequencies = np.load('cubi_frequencies.npy')
    multi_xs = np.load('multi_xs.npy')
    multi_ys = np.load('multi_ys.npy')
    multi_weights = np.load('multi_weights.npy')
    multi_frequencies = np.load('multi_frequencies.npy')

    plot_r3d = False
    plot_cubi = True
    plot_multi = True
    if plot_r3d:
        type = 'r3d'
        ticks = ['background', 'walls', 'openings']
        multi_r3d_mapping = [0, 1, 1, 1, 2, 2, 2, None]

        cubi_weights = map_weights(cubi_weights, multi_r3d_mapping, ticks)
        multi_weights = map_weights(multi_weights, multi_r3d_mapping, ticks)

        cubi_frequencies = map_frequencies(cubi_frequencies, multi_r3d_mapping, ticks)
        multi_frequencies = map_frequencies(multi_frequencies, multi_r3d_mapping, ticks)

        labels = ['R3D', 'CubiCasa', 'MUFP']
        bin_dims(data=[(r3d_xs, r3d_ys), (cubi_xs, cubi_ys), (multi_xs, multi_ys)], labels=labels, type=type)
        plot_areas([r3d_weights, cubi_weights, multi_weights], ticks=ticks, labels=labels, type=type)
        plot_frequencies([r3d_frequencies, cubi_frequencies, multi_frequencies], ticks=ticks, labels=labels, type=type)
    elif plot_cubi:
        type = 'cubi'
        ticks = ['background', 'walls', 'railings', 'doors', 'windows', 'stairs']
        multi_cubi_mapping = [0, 1, 1, 2, 3, 3, 4, 5]

        cubi_weights = map_weights(cubi_weights, multi_cubi_mapping, ticks)
        multi_weights = map_weights(multi_weights, multi_cubi_mapping, ticks)

        cubi_frequencies = map_frequencies(cubi_frequencies, multi_cubi_mapping, ticks)
        multi_frequencies = map_frequencies(multi_frequencies, multi_cubi_mapping, ticks)

        labels = ['CubiCasa', 'MUFP']
        bin_dims(data=[(cubi_xs, cubi_ys), (multi_xs, multi_ys)], labels=labels, type=type)
        # plot_areas([cubi_weights, multi_weights], ticks=ticks, labels=labels, type=type)
        # plot_frequencies([cubi_frequencies, multi_frequencies], ticks=ticks, labels=labels, type=type)
    elif plot_multi:
        type = 'multi'
        ticks = ['background', 'walls', 'glass walls', 'railings', 'doors', 'sliding doors', 'windows', 'stairs']

        labels = ['MUFP']
        bin_dims(data=[(multi_xs, multi_ys)], labels=labels, type=type)
        plot_areas([multi_weights], ticks=ticks, labels=labels, type=type)
        plot_frequencies([multi_frequencies], ticks=ticks, labels=labels, type=type)
