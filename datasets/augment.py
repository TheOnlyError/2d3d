import os
import random
from functools import partial
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image

from training import utils

Image.MAX_IMAGE_PIXELS = None

RESIZE_MODE = 'resize'  # pad
MIN_SIZE = 512
MAX_SIZE = 832
STRIDE = 64

img_sizes = {
    'multi_plans/image0/image.jpg': 2000,
    'multi_plans/image1/image.jpg': 2000,
    'multi_plans/image2/image.jpg': 2000,
    'multi_plans/image3/image.jpg': 2400,
    'multi_plans/image4/image.jpg': 2800,
    'multi_plans/image5/image.jpg': 2400,
    'multi_plans/image6/image.jpg': 2800,
    'multi_plans/image7/image.jpg': 2200,
}


def rotate_image(mat, angle, val):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
        width / 2,
        height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h), borderValue=val, flags=cv2.INTER_NEAREST)
    return rotated_mat


def generate_samples(input, mask, sizes, file):
    samples = []

    w = img_sizes.get(file, 2000)
    if input.shape[1] > w:
        h = int(w / input.shape[1] * input.shape[0])
        input = cv2.resize(input, (w, h), interpolation=cv2.INTER_AREA)

    h, w, _ = input.shape
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    prediction_count = np.zeros((h, w))

    only_resize = True
    if only_resize:
        sample = ['[]']
        max_d = max(h, w)
        max_d_pad = max_d + STRIDE - max_d % STRIDE
        sample += pad_square(input, mask, h, w, max_d_pad)
        return [sample]

    # First stage: random sample
    size = sizes[-1]
    hsize = int(size / 2)
    points = zip(np.random.randint(0, h, size=1), np.random.randint(0, w, size=1))
    for p in points:
        y = p[0]
        x = p[1]
        y1 = max(0, y - hsize + min(0, h - (y + hsize)))
        y2 = min(h, y + hsize + max(0, -(y - hsize)))
        x1 = max(0, x - hsize + min(0, w - (x + hsize)))
        x2 = min(w, x + hsize + max(0, -(x - hsize)))
        prediction_count[y1:y2, x1:x2] += 1
        samples.append([y1, y2, x1, x2])

    # Second stage: 1 pass over matrix
    uncovered = np.argwhere(prediction_count == 0)
    while uncovered.any():
        points = uncovered[np.random.choice(len(uncovered), size=min(len(uncovered), 1), replace=False)]
        size = sizes[-1]
        hsize = int(size / 2)
        for p in points:
            y = p[0]
            x = p[1]
            y1 = max(0, y - hsize + min(0, h - (y + hsize)))
            y2 = min(h, y + hsize + max(0, -(y - hsize)))
            x1 = max(0, x - hsize + min(0, w - (x + hsize)))
            x2 = min(w, x + hsize + max(0, -(x - hsize)))
            prediction_count[y1:y2, x1:x2] += 1
            samples.append([y1, y2, x1, x2])

        uncovered = np.argwhere(prediction_count == 0)

    resize_samples = []
    for s in samples:
        y1, y2, x1, x2 = s
        sample = input[y1:y2, x1:x2]
        # Only append samples that are non empty
        # TODO check if this works for white images
        if (sample != 255).any():
            size = random.choice(sizes)
            sample_resize = cv2.resize(input[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA)
            mask_resize = cv2.resize(mask[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_NEAREST)
            resize_samples.append([str(s), sample_resize, mask_resize])
    return resize_samples


def pad(input, mask, h, w, new_h, new_w):
    lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)

    enlarge = (mask.shape[0] * mask.shape[1]) > (h * w)
    interpolation = cv2.INTER_CUBIC if enlarge else cv2.INTER_AREA
    input = cv2.resize(input, (w, h), interpolation=interpolation)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    input = np.pad(input, [(lh, uh), (lw, uw), (0, 0)], mode='constant', constant_values=255)
    mask = np.pad(mask, [(lh, uh), (lw, uw)], mode='constant', constant_values=0)

    return input, mask


def pad_square(input, mask, h, w, new_d):
    lh, uh = int((new_d - h) / 2), int(new_d - h) - int((new_d - h) / 2)
    lw, uw = int((new_d - w) / 2), int(new_d - w) - int((new_d - w) / 2)

    enlarge = (input.shape[0] * input.shape[1]) < (h * w)
    interpolation = cv2.INTER_CUBIC if enlarge else cv2.INTER_AREA
    input = cv2.resize(input, (w, h), interpolation=interpolation)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    input = np.pad(input, [(lh, uh), (lw, uw), (0, 0)], mode='constant', constant_values=255)
    mask = np.pad(mask, [(lh, uh), (lw, uw)], mode='constant', constant_values=0)

    return input, mask


def resize(input, mask, sizes, file):
    min_size = sizes[0]
    max_size = sizes[-1]
    max_pixels = max_size ** 2
    h, w, _ = input.shape
    max_d = max(h, w)
    min_d = min(h, w)
    aspect_ratio = min_d / max_d
    max_d_pad = max_d + STRIDE - max_d % STRIDE
    min_d_pad = min_d + STRIDE - min_d % STRIDE
    if max_d_pad * min_d_pad > 1.5 * max_pixels:
        # Apply sampling if image is too large
        return generate_samples(input, mask, sizes, file)
    else:
        if RESIZE_MODE == 'resize' or RESIZE_MODE == 'pad':
            if RESIZE_MODE == 'resize':
                max_index = int(max_size / STRIDE) - int(min_size / STRIDE)
                max_d = sizes[min(max_index, max(0, int(max_d / STRIDE) - int(min_size / STRIDE)))]
                max_d_pad = max_d
                min_d = round(aspect_ratio * max_d)

            # min_d_pad = min_d + STRIDE - min_d % STRIDE
            sample = ['[]']
            if h == max(h, w):
                sample += pad_square(input, mask, max_d, min_d, max_d_pad)
            else:
                sample += pad_square(input, mask, min_d, max_d, max_d_pad)
            return [sample]
        else:
            raise Exception('Invalid RESIZE_MODE: {0}'.format(RESIZE_MODE))


def augmentation(dataset):
    print("Starting augmentation")

    sizes = list(range(MIN_SIZE, MAX_SIZE + STRIDE, STRIDE))

    train_paths = open(dataset + '_train.txt', 'r').read().splitlines()
    val_paths = open(dataset + '_val.txt', 'r').read().splitlines()
    test_paths = open(dataset + '_test.txt', 'r').read().splitlines()
    paths = train_paths + val_paths + test_paths
    print(len(paths))

    pool = Pool(processes=10)
    pool.map(partial(process, dataset, sizes), paths)
    pool.close()
    pool.join()


def process(dataset, sizes, p):
    base = 'annotations/hdd/'
    path = 'annotations/hdd/' + dataset
    new_path = path + '_augment/'

    i = p.split('\t')
    n = i[0].split('/')[1]
    input_org = cv2.imread(base + i[0])
    mask_org = cv2.imread(base + i[1], 0)

    # Produce flips, rotations, resize
    for hflip in [False, True]:
        input_hflip = np.copy(input_org)
        mask_hflip = np.copy(mask_org)

        # Vertical flip
        if hflip:
            input_hflip = cv2.flip(input_hflip, 0)
            mask_hflip = cv2.flip(mask_hflip, 0)

        for rotate in [False, True]:
            augments = '_'.join([str(hflip), str(rotate)])

            input_rotate = np.copy(input_hflip)
            mask_rotate = np.copy(mask_hflip)

            if rotate:
                angle = int(random.uniform(-90, 90))
                input_rotate = rotate_image(input_rotate, angle, (255, 255, 255))
                mask_rotate = rotate_image(mask_rotate, angle, (0, 0, 0))

            # Resize
            samples = resize(input_rotate, mask_rotate, sizes, i[0])

            # Save augmented images
            for name, input, mask in samples:
                os.mkdir(new_path + n + '_' + augments + '_' + name + '/')
                cv2.imwrite(new_path + n + '_' + augments + '_' + name + '/input.png', input)
                cv2.imwrite(new_path + n + '_' + augments + '_' + name + '/mask.png', mask)
                cv2.imwrite(new_path + n + '_' + augments + '_' + name + '/mask_rgb.png', utils.ind2rgb(mask))


def main():
    # Initialize seeds
    seed = int(random.random() * 1000)
    np.random.seed(seed)
    random.seed(seed)
    print("Seed: {0}".format(seed))

    # augmentation('r3d')
    # augmentation('cubicasa5k')
    # augmentation('multi_plans')
    augmentation('multi_plans_test')

    # sampling()
    # augmentation('multi_plans_sampled')


if __name__ == '__main__':
    main()
