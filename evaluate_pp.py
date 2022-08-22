import logging
import os
import time
from functools import partial
from multiprocessing.pool import Pool, ThreadPool

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

from training.confusion_matrix import CM

import segmentation_models as sm
from datasets import floorplans
from training.metrics import cm_metrics
from training.post_process import post_process

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
logging.disable(logging.WARNING)


def process(conversion, i_result):
    i, result = i_result
    y_true, y_pred = result
    if i > 90000:
        print('Done: ' + str(i))
        return y_true, y_pred
    if conversion is not None:
        converted_ind = np.zeros(y_pred.shape, dtype=np.uint8)
        if conversion == 'r3d':
            converted_ind[y_pred == 1] = 1
            converted_ind[y_pred == 2] = 4
        elif conversion == 'cubicasa5k':
            converted_ind[y_pred == 1] = 1
            converted_ind[y_pred == 2] = 3
            converted_ind[y_pred == 3] = 4
            converted_ind[y_pred == 4] = 6
            converted_ind[y_pred == 5] = 7
        y_pred = converted_ind

    y_pred = y_pred.astype(np.uint8)
    y_pred = post_process(y_pred, high_res=True, zeng=False)

    if conversion is not None:
        converted_ind = np.zeros(y_pred.shape, dtype=np.uint8)
        if conversion == 'r3d':
            converted_ind[y_pred == 1] = 1
            converted_ind[y_pred == 4] = 2
        elif conversion == 'cubicasa5k':
            converted_ind[y_pred == 1] = 1
            converted_ind[y_pred == 3] = 2
            converted_ind[y_pred == 4] = 3
            converted_ind[y_pred == 6] = 4
            converted_ind[y_pred == 7] = 5
        y_pred = converted_ind

    print('Done: ' + str(i))
    return y_true, y_pred


def main():
    sm.set_framework('tf.keras')

    configs = [

    ]

    for config in configs:
        conversion = None
        datasets = config['datasets']
        if len(datasets) == 1:
            if datasets[0] == 'r3d_augment':
                conversion = 'r3d'
            elif datasets[0] == 'cubicasa5k_augment':
                conversion = 'cubicasa5k'

        classes = ['bg'] + config['classes']
        tta = 'tta' in config and config['tta']
        post_processing = True
        if tta:
            print('Fix this!')
            exit(0)

        identifier = 'results/hdd/' + config['model'] + '_' + config.get('name', '') + '_' + str(tta) + '_' + str(
            post_processing)

        if os.path.exists(identifier + '.npy'):
            results = np.load(identifier + '.npy', allow_pickle=True)
            cm_obj = CM(num_classes=len(classes), post_processing=post_processing, conversion=conversion)

            mp = True
            if mp:
                pool = Pool(processes=60)
                pp_results = [x for x in pool.imap_unordered(partial(process, conversion), enumerate(results))]
                pool.close()
                pool.join()
            else:
                pp_results = []
                for result in results:
                    pp_results.append(process(conversion, result))

            print(len(pp_results))
            for (y_true, y_pred) in pp_results:
                cm_obj.update_state(y_true, y_pred)
            cm = cm_obj.result().numpy()
            print(cm)
            timestr = time.strftime("%Y%m%d-%H%M%S")

            np.savetxt(
                'results/cm' + '_' + config['model'] + '_' + config.get('name', '') + '_' + str(tta) + '_' + str(
                    post_processing) + '_' + timestr + '.txt',
                cm)

            # Plot confusion matrices
            plot_cm = False
            if plot_cm:
                cms = [cm, cm / cm.sum(axis=0, keepdims=True), cm / cm.sum(axis=1, keepdims=True)]
                titles = [
                    'Original',
                    'What classes are responsible for each classification',
                    'How each class has been classified'
                ]
                for t, confusion_matrix in enumerate(cms):
                    fig, ax = plt.subplots()
                    ax.matshow(confusion_matrix)
                    ax.set_title(titles[t])
                    ax.set_xlabel('Predicted class')
                    ax.set_ylabel('True class')
                    ax.xaxis.set_ticks_position('bottom')
                    ax.set_xticks(list(range(len(classes))))
                    ax.set_yticks(list(range(len(classes))))
                    ax.set_xticklabels(classes, rotation='vertical')
                    ax.set_yticklabels(classes)
                    for i in range(len(classes)):
                        for j in range(len(classes)):
                            c = round(confusion_matrix[j, i], 2)
                            ax.text(i, j, str(c), va='center', ha='center')
                    plt.savefig('results/cm' + str(t) + '_' + config['model'] + '_' + str(tta) + '_' + str(
                        post_processing) + '_' + timestr, bbox_inches='tight')
                    plt.show()

            # Compute metrics
            f = open(
                'results/metrics' + '_' + config['model'] + '_' + config.get('name', '') + '_' + str(tta) + '_' + str(
                    post_processing) + '_' + timestr + '.txt', 'w')
            table = []
            columns = cm_metrics(cm)

            table.append(['Accuracy', np.diag(cm).sum() / cm.sum()])
            cm_nobg = np.copy(cm)
            cm_nobg[0] = 0
            table.append(['Accuracy no bg', np.diag(cm_nobg).sum() / cm_nobg.sum()])
            for i, c in enumerate(classes):
                table.append(
                    [c] + [column[i] for column in columns[:-1]] + [
                        np.array([metrics[i] for metrics in columns[-1]]) / sum(
                            [metrics[i] for metrics in columns[-1]])])
            table.append(['Mean'] + [np.nanmean(column) for column in columns[:-1]] + [
                np.array(np.sum(columns[-1], axis=1)) / np.sum(columns[-1], axis=1).sum()])
            table.append(['Mean no bg'] + [np.nanmean(column[1:]) for column in columns[:-1]] + [
                np.array(np.sum(columns[-1][1:], axis=1)) / np.sum(columns[-1][1:], axis=1).sum()])
            f.write(
                tabulate(table,
                         headers=['Class', 'Class accuracy', 'Recall', 'Precision', 'F1', 'IoU', 'fwRecall', 'fwIoU',
                                  'TP, FP, TN, FN']))
            f.close()
        else:
            unet_model = tf.keras.models.load_model('models/' + config['model'], compile=False)
            unet_model.compile(run_eagerly=True)

            # NORMALIZE=False IF EFFICIENTNET IS USED
            # Otherwise=True
            backbone = config['backbone']
            normalize = backbone[0] != 'e'
            print("Backbone: {0}, normalize: {1}".format(backbone, normalize))
            test_dataset = floorplans.load_test_data(classes, config['datasets'], normalize=normalize)

            results = []
            for (image, y_true) in test_dataset.batch(1):
                y_pred = unet_model.predict_on_batch(image)
                if isinstance(y_pred, tuple):
                    y_pred = y_pred[1]
                y_pred = y_pred[0].argmax(axis=-1)
                results.append((y_true, y_pred))

            del unet_model
            tf.keras.backend.clear_session()
            print('========== DONE ==========')
            print(len(results))

            np.save(identifier, results)


if __name__ == "__main__":
    tic = time.time()
    main()
    toc = time.time()
    print('total evaluation time = {} minutes'.format((toc - tic) / 60))
