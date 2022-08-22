import numpy as np


def cm_metrics(confusion_matrix):
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)

    # Class accuracy
    acc = (TP + TN) / (TP + FP + FN + TN)

    # Sensitivity, hit rate, recall, or true positive rate
    recall = TP / (TP + FN)

    # Precision or positive predictive value
    precision = TP / (TP + FP)

    f1 = 2 * precision * recall / (precision + recall)

    # IoU
    iou = TP / (TP + FN + FP)

    # fw
    fw = TP / (np.diag(confusion_matrix).sum() - confusion_matrix[0, 0])

    fwIou = fw * iou

    return acc, recall, precision, f1, iou, fw, fwIou, [TP, FP, TN, FN]


if __name__ == '__main__':
    cm = np.loadtxt(
        '../results/cm_unet3p_heatmap_aaf2_EfficientNetB2_64,128,256,512,1024_multi_plans_augment_20220726-185321_heatmap_aaf2_False_False_20220726-214152.txt')
    metrics = cm_metrics(cm)
    print(metrics[1][1:])
    print(metrics[4][1:])
