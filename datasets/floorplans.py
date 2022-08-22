import random

from training import utils
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

dataset_classes = []


def load_train_data(classes, datasets, normalize=True, buffer_size=400) -> [tf.data.Dataset, tf.data.Dataset,
                                                                            tf.data.Dataset]:
    global dataset_classes
    dataset_classes = classes
    dataset = '_'.join(datasets)
    train_dataset = load_dataset(dataset + '_train', normalize).shuffle(buffer_size)
    # train_dataset = load_dataset(dataset + '_train', normalize).take(2000).shuffle(buffer_size)
    # train_dataset = load_dataset(dataset + '_train', normalize).shuffle(buffer_size)
    val_dataset = load_dataset(dataset + '_val', normalize)
    test_dataset = load_dataset(dataset + '_test', normalize)
    return train_dataset, val_dataset, test_dataset


def load_test_data(classes, datasets, normalize=True) -> [tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    global dataset_classes
    dataset_classes = classes
    dataset = '_'.join(datasets)
    test_dataset = load_dataset(dataset + '_test', normalize)
    return test_dataset


def _parse_function(example_proto):
    feature = {'image': tf.io.FixedLenFeature([], tf.string),
               'mask': tf.io.FixedLenFeature([], tf.string),
               'heatmaps': tf.io.FixedLenFeature([], tf.string),
               'width': tf.io.FixedLenFeature([], tf.int64),
               'height': tf.io.FixedLenFeature([], tf.int64)}
    example = tf.io.parse_single_example(example_proto, feature)

    image, mask, heatmaps, width, height = decode_all_raw(example)
    return preprocess(image, mask, heatmaps, width, height)


def _parse_function_normalize(example_proto):
    feature = {'image': tf.io.FixedLenFeature([], tf.string),
               'mask': tf.io.FixedLenFeature([], tf.string),
               'heatmaps': tf.io.FixedLenFeature([], tf.string),
               'width': tf.io.FixedLenFeature([], tf.int64),
               'height': tf.io.FixedLenFeature([], tf.int64)}
    example = tf.io.parse_single_example(example_proto, feature)

    image, mask, heatmaps, width, height = decode_all_raw(example)
    return preprocess_normalize(image, mask, heatmaps, width, height)


def decode_all_raw(x):
    image = tf.io.decode_raw(x['image'], tf.uint8)
    mask = tf.io.decode_raw(x['mask'], tf.uint8)
    heatmaps = tf.io.decode_raw(x['heatmaps'], tf.uint8)
    width = tf.cast(x['width'], tf.int32)
    height = tf.cast(x['height'], tf.int32)
    return image, mask, heatmaps, width, height


def preprocess(img, mask, heatmaps, width, height):
    global dataset_classes
    img = tf.cast(img, dtype=tf.float32)
    img = tf.reshape(img, [height, width, 3])
    mask = tf.reshape(mask, [height, width])
    heatmaps = tf.cast(heatmaps, dtype=tf.float32)
    heatmaps = tf.reshape(heatmaps, [height, width, -1]) / 255

    mask = tf.one_hot(mask, depth=len(dataset_classes))
    mask = tf.concat([mask, heatmaps], axis=-1)

    return img, mask


def preprocess_normalize(img, mask, heatmaps, width, height):
    global dataset_classes
    img = tf.cast(img, dtype=tf.float32)
    img = tf.reshape(img, [height, width, 3]) / 255
    mask = tf.reshape(mask, [height, width])
    heatmaps = tf.cast(heatmaps, dtype=tf.float32)
    heatmaps = tf.reshape(heatmaps, [height, width, -1]) / 255

    mask = tf.one_hot(mask, depth=len(dataset_classes))
    mask = tf.concat([mask, heatmaps], axis=-1)

    return img, mask


def load_dataset(type, normalize):
    raw_dataset = tf.data.TFRecordDataset(type + '.tfrecords')
    if normalize:
        parsed_dataset = raw_dataset.map(_parse_function_normalize)
    else:
        parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset


def create_dataset(datasets, type='default'):
    print('Creating dataset')

    train_paths = []
    val_paths = []
    test_paths = []
    for i in datasets:
        train_paths += open(i + '_train.txt', 'r').read().splitlines()
        val_paths += open(i + '_val.txt', 'r').read().splitlines()
        test_paths += open(i + '_test.txt', 'r').read().splitlines()

    random.shuffle(train_paths)
    random.shuffle(val_paths)
    random.shuffle(test_paths)

    # test_paths = [p for p in test_paths if 'True' not in p[0]]
    # print(len(test_paths))

    dataset = '_'.join(datasets)
    write_record(train_paths, name=dataset + '_train', type=type)
    write_record(val_paths, name=dataset + '_val', type=type)
    write_record(test_paths, name=dataset + '_test', type=type)


def write_record(paths, name, type):
    writer = tf.io.TFRecordWriter(name + '.tfrecords')

    print(len(paths))
    for i in range(len(paths)):
        # Load the image
        image, mask, heatmaps = load_images(paths[i], type)
        if type == 'cubicasa5k':
            # Compress labels
            # From  ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all']
            # To    ['walls', 'railings', 'doors', 'windows', 'stairs_all']
            mask_temp = np.copy(mask)
            mask[(mask_temp == 3)] = 2
            mask[(mask_temp == 4)] = 3
            mask[(mask_temp == 6)] = 4
            mask[(mask_temp == 7)] = 5

        h, w = mask.shape

        # Create a feature
        feature = {'image': _bytes_feature(tf.compat.as_bytes(image.tostring())),
                   'mask': _bytes_feature(tf.compat.as_bytes(mask.tostring())),
                   'heatmaps': _bytes_feature(tf.compat.as_bytes(heatmaps.tostring())),
                   'width': _int64_feature(w),
                   'height': _int64_feature(h)}

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()


def load_images(path, type=''):
    base = 'annotations/hdd/'
    images = path.split('\t')

    image = cv2.imread(base + images[0]).astype(np.uint8)
    mask = cv2.imread(base + images[1], 0).astype(np.uint8)

    # image = cv2.resize(image, (512, 512))  # TODO check augment for proper interpolation
    # image = cv2.resize(image, (608, 608))  # TODO check augment for proper interpolation
    # image = cv2.resize(image, (2048, 2048))  # TODO check augment for proper interpolation
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    if type == 'combi' and len(images[2:]) < 3:
        # classes = ['walls', 'railings', 'doors', 'windows', 'stairs_all']
        # classes = ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all']
        # Map heatmaps so that they are compatible
        # Insert empty heatmap at second index for missing sliding doors
        heatmaps = [
            cv2.resize(cv2.imread(base + images[2], 0).astype(np.uint8), (image.shape[1], image.shape[0])),
            np.zeros((image.shape[0], image.shape[1]), np.uint8),
            cv2.resize(cv2.imread(base + images[3], 0).astype(np.uint8), (image.shape[1], image.shape[0])),
        ]
    else:
        heatmaps = []
        for heatmap in images[2:]:
            heatmaps.append(
                cv2.resize(cv2.imread(base + heatmap, 0).astype(np.uint8), (image.shape[1], image.shape[0])))
        if len(heatmaps) == 0:
            heatmaps = [np.zeros(mask.shape)]

    heatmaps = np.stack(heatmaps, axis=-1)
    print(heatmaps.shape)

    show = False
    if show:
        print(images[0].split('/')[1])
        plt.figure(dpi=400)
        plt.imshow(np.array(image).astype(np.uint8))
        # print(np.unique(np.array(tf.math.argmax(mask[0], axis=-1))))
        # plt.imshow(ind2rgb(mask), alpha=0.6)
        plt.imshow(utils.ind2rgb(mask), alpha=0.6)
        plt.savefig('temp/' + images[0].split('/')[1])
        # plt.show()
        plt.close()
        plt.figure(dpi=400)
        plt.imshow(np.array(image).astype(np.uint8))
        # print(np.unique(np.array(tf.math.argmax(mask[0], axis=-1))))
        # plt.imshow(ind2rgb(mask), alpha=0.6)
        # plt.savefig('temp/' + images[0].split('/')[1] + '_org')
        plt.show()
        plt.close()

    return image, mask, heatmaps


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


if __name__ == "__main__":
    datasets = [
        # 'r3d_augment',
        'cubicasa5k_augment',
        'multi_plans_augment',
        # 'multi_plans_test_augment',
    ]
    # type = ''
    # type = 'cubicasa5k'
    type = 'combi'
    show = False
    if not show:
        create_dataset(datasets, type=type)
    else:
        from training import utils

        # classes = ['walls', 'openings']
        # classes = ['walls', 'railings', 'doors', 'windows', 'stairs_all']
        # classes = ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows']
        classes = ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all']
        classes = ['bg'] + classes

        # dataset = load_train_data(classes, datasets, normalize=False)[0]
        dataset = load_test_data(classes, datasets, normalize=False)
        # for (image, mask) in dataset.shuffle(32).take(2).batch(1):
        for (image, mask) in dataset.take(2).batch(1):
            heatmaps = mask[:, :, :, len(classes):]
            mask = mask[:, :, :, :len(classes)]
            mask = np.array(tf.math.argmax(mask[0], axis=-1))
            showMask = True
            if showMask:
                # print(np.unique(np.array(image[0])))
                plt.figure(dpi=400)
                print(image[0].shape)
                plt.imshow(np.array(image[0]).astype(np.uint8))
                # print(np.unique(np.array(tf.math.argmax(mask[0], axis=-1))))
                print(np.unique(mask))
                plt.imshow(utils.ind2rgba(mask), alpha=0.8)
                plt.show()
                plt.close()
            else:
                # mask = utils.ind2rgb(mask)
                heatmap = heatmaps[0, :, :, 0].numpy()
                print(np.unique(heatmap))
                plt.imshow(heatmap)
                # plt.imshow(cv2.imread('annotations/hdd/r3d_augment/45741076.jpg_False_False_[]/heatmap_openings.png', 0), cmap='gray')
                # plt.imshow(cv2.imread('annotations/hdd/r3d_augment/45741076.jpg_False_False_[]/heatmap_rgb_openings.png', 0), cmap='gray')
                plt.show()
                plt.close()
