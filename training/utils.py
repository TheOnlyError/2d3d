from typing import Tuple
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def crop_to_shape(data, shape: Tuple[int, int, int]):
    """
    Crops the array to the given image shape by removing the border

    :param data: the array to crop, expects a tensor of shape [batches, nx, ny, channels]
    :param shape: the target shape [batches, nx, ny, channels]
    """
    diff_nx = (data.shape[0] - shape[0])
    diff_ny = (data.shape[1] - shape[1])

    if diff_nx == 0 and diff_ny == 0:
        return data

    offset_nx_left = diff_nx // 2
    offset_nx_right = diff_nx - offset_nx_left
    offset_ny_left = diff_ny // 2
    offset_ny_right = diff_ny - offset_ny_left

    cropped = data[offset_nx_left:(-offset_nx_right), offset_ny_left:(-offset_ny_right)]

    assert cropped.shape[0] == shape[0]
    assert cropped.shape[1] == shape[1]
    return cropped


def crop_labels_to_shape(shape: Tuple[int, int, int]):
    def crop(image, label):
        return image, crop_to_shape(label, shape)

    return crop


def crop_image_and_label_to_shape(shape: Tuple[int, int, int]):
    def crop(image, label):
        return crop_to_shape(image, shape), \
               crop_to_shape(label, shape)

    return crop


def to_rgb(img: np.array):
    """
    Converts the given array into a RGB image and normalizes the values to [0, 1).
    If the number of channels is less than 3, the array is tiled such that it has 3 channels.
    If the number of channels is greater than 3, only the first 3 channels are used

    :param img: the array to convert [bs, nx, ny, channels]

    :returns img: the rgb image [bs, nx, ny, 3]
    """
    img = img.astype(np.float32)
    img = np.atleast_3d(img)

    channels = img.shape[-1]
    if channels == 1:
        img = np.tile(img, 3)

    elif channels == 2:
        img = np.concatenate((img, img[..., :1]), axis=-1)

    elif channels > 3:
        img = img[..., :3]

    img[np.isnan(img)] = 0
    img -= np.amin(img)
    if np.amax(img) != 0:
        img /= np.amax(img)

    return img


# classes = ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all']
# floorplan_map = {
#     0: [0, 0, 0],  # background white
#     1: [255, 255, 255],  # walls black
#     2: [230, 25, 75],  # glass_walls red
#     3: [60, 180, 75],  # railings green
#     4: [255, 225, 25],  # doors yellow
#     5: [0, 130, 200],  # sliding_doors blue
#     6: [245, 130, 48],  # windows orange
#     7: [70, 240, 240],  # stairs_all cyan
# }

door_map = {
    4: [255, 225, 25],  # doors yellow
}

floorplan_map = {
    0: [255, 255, 255],  # background white
    1: [0, 0, 0],  # walls black
    2: [230, 25, 75],  # glass_walls red
    3: [60, 180, 75],  # railings green
    4: [255, 225, 25],  # doors yellow
    5: [0, 130, 200],  # sliding_doors blue
    6: [245, 130, 48],  # windows orange
    7: [70, 240, 240],  # stairs_all cyan
}

floorplan_map_rgba = {
    0: [255, 255, 255, 0],  # background white
    1: [0, 0, 0, 255],  # walls black
    2: [230, 25, 75, 255],  # glass_walls red
    3: [60, 180, 75, 255],  # railings green
    4: [255, 225, 25, 255],  # doors yellow
    5: [0, 130, 200, 255],  # sliding_doors blue
    6: [245, 130, 48, 255],  # windows orange
    7: [70, 240, 240, 255],  # stairs_all cyan
}


def ind2rgb(ind_img, color_map=None, conversion=None):
    if color_map is None:
        color_map = floorplan_map
    rgb_img = np.zeros((ind_img.shape[0], ind_img.shape[1], 3), dtype=np.uint8)

    if conversion is not None:
        # classes = ['walls', 'openings']
        # classes = ['walls', 'railings', 'doors', 'windows', 'stairs_all']
        # classes = ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all']
        converted_ind = np.zeros(ind_img.shape, dtype=np.uint8)
        if conversion == 'r3d':
            converted_ind[ind_img == 1] = 1
            converted_ind[ind_img == 2] = 4
        elif conversion == 'cubicasa5k':
            converted_ind[ind_img == 1] = 1
            converted_ind[ind_img == 2] = 3
            converted_ind[ind_img == 3] = 4
            converted_ind[ind_img == 4] = 6
            converted_ind[ind_img == 5] = 7
        ind_img = converted_ind

    for i, rgb in color_map.items():
        rgb_img[(ind_img == i)] = rgb

    return rgb_img


def ind2rgba(ind_img, color_map=None):
    if color_map is None:
        color_map = floorplan_map_rgba
    rgb_img = np.zeros((ind_img.shape[0], ind_img.shape[1], 4), dtype=np.uint8)

    for i, rgb in color_map.items():
        rgb_img[(ind_img == i)] = rgb

    return rgb_img


def rgb2ind(img, color_map=None):
    if color_map is None:
        color_map = floorplan_map
    ind = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    for i, rgb in color_map.items():
        ind[(img == rgb).all(2)] = i

    return ind


def plot_legend():
    classes = ['background', 'walls', 'glass walls', 'railings', 'doors', 'sliding doors', 'windows', 'stairs']

    fig, ax = plt.subplots(dpi=400)
    patches = []
    for i, c in enumerate(classes):
        patches.append(mpatches.Patch(color=np.array(floorplan_map.get(i)) / 255, label=c[0].upper() + c[1:]))
    ax.legend(handles=patches)
    # ax.legend(handles=patches, ncol=len(classes))
    plt.savefig('legend', bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    plot_legend()
