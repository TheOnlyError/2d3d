import random
import os

heatmap_classes = {
    'r3d_augment': ['openings'],
    'cubicasa5k_augment': ['doors', 'windows'],
    'multi_plans_augment': ['doors', 'sliding_doors', 'windows'],
}


def split(dataset, input_file='input.png', test_only=False):
    heatmaps = heatmap_classes.get(dataset, [])

    base = dataset + '/'
    dirs = os.listdir('annotations/hdd/' + base)
    random.shuffle(dirs)

    if test_only:
        n_train = 0
        n_val = 0
        n_test = len(dirs)
    else:
        n_train = int(0.6 * len(dirs))
        n_val = int((len(dirs) - n_train) / 2)
        n_test = len(dirs) - n_train - n_val

    print("Total: {0}, train: {1}, val: {2}, test: {3}".format(len(dirs), n_train, n_val, n_test))

    train = dirs[:n_train]
    val = dirs[n_train:n_train + n_val]
    test = dirs[n_train + n_val:]

    f = open(dataset + '_train.txt', 'w')
    for i in train:
        input = base + i + '/' + input_file
        mask = base + i + '/mask.png'
        line = input + '\t' + mask
        for h in heatmaps:
            line += '\t' + base + i + '/heatmap_' + h + '.png'
        f.write(line + '\n')
    f.close()

    f = open(dataset + '_val.txt', 'w')
    for i in val:
        input = base + i + '/' + input_file
        mask = base + i + '/mask.png'
        line = input + '\t' + mask
        for h in heatmaps:
            line += '\t' + base + i + '/heatmap_' + h + '.png'
        f.write(line + '\n')
    f.close()

    f = open(dataset + '_test.txt', 'w')
    for i in test:
        input = base + i + '/' + input_file
        mask = base + i + '/mask.png'
        line = input + '\t' + mask
        for h in heatmaps:
            line += '\t' + base + i + '/heatmap_' + h + '.png'
        f.write(line + '\n')
    f.close()


if __name__ == '__main__':
    # split('cubicasa5k_single_augment')
    split('cubicasa5k_augment')
    # split('multi_plans', 'image.jpg')
    # split('multi_plans_augment')
    # split('multi_plans_test', 'image.jpg', True)
    # split('multi_plans_test_augment', test_only=True)
    split('r3d_augment')
