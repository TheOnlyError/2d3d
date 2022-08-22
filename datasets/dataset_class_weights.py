import cv2
import numpy as np



weights = {
    'r3d_augment': [0.00713544, 0.1628326, 0.83003196],
    'cubicasa5k_single_augment': [0.02306051, 0.41013369, 0.24201805, 0.10743052, 0.21735723]
}


def weigh_dataset(datasets):
    print('Weighing dataset')

    paths = []
    for i in datasets:
        paths += open(i + '_train.txt', 'r').read().splitlines()
        paths += open(i + '_val.txt', 'r').read().splitlines()
        paths += open(i + '_test.txt', 'r').read().splitlines()

    print(len(paths))
    weights = [0 for _ in range(len(classes))]
    for path in paths:
        mask = load_images(path)
        # mask_temp = np.copy(mask)
        # mask[(mask_temp == 3)] = 2
        # mask[(mask_temp == 4)] = 3
        # mask[(mask_temp == 6)] = 4
        # mask[(mask_temp == 7)] = 5
        for c in range(len(classes)):
            weights[c] += np.count_nonzero(mask == c)
    return weights


def load_images(path):
    base = 'annotations/hdd/'
    images = path.split('\t')

    mask = cv2.imread(base + images[1], 0).astype(np.uint8)

    return mask

def smooth(vals):
    vals = np.array([0.00129272, 0.03085887, 0.52862675, 0.2947635, 0.14445817])
    for i in range(1):
        vals += 1/len(vals)/10
        vals /= sum(vals)
        print(vals)


if __name__ == "__main__":
    # smooth([])
    # exit()


    # classes = ['walls', 'openings']
    # classes = ['walls', 'railings', 'doors', 'windows', 'stairs_all']
    classes = ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows']
    # classes = ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all']
    classes = ['bg'] + classes
    datasets = [
        # 'r3d_augment'
        'cubicasa5k_single_augment',
        # 'cubicasa5k_augment',
        'multi_plans_sampled_augment'

    ]
    weights = weigh_dataset(datasets)
    print(weights)

    # weights = [2203905105, 92960765, 1792478, 5608471, 9445117, 491882, 19014518, 8413664]

    # With bg class
    # weights_bg = sum(weights) / np.array(weights)
    # weights_bg = weights_bg / sum(weights_bg)
    # print(weights_bg)

    # Remove bg class
    classes = classes[1:]
    weights = weights[1:]
    print('=========== Percentages ===========')
    print(*list(zip(classes, np.array(weights) / sum(weights))), sep='\n')
    weights = sum(weights) / np.array(weights)
    weights = weights / sum(weights)
    print('=========== Weights ===========')
    print(*list(zip(classes, weights)), sep='\n')
