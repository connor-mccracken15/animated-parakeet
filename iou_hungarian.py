"""
IoU + Hungarian linking between two label masks.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

def iou_matrix(mask_a, mask_b, labels_a, labels_b):

    iou = np.zeros((len(labels_a), len(labels_b)))

    for i, label_a in enumerate(labels_a):
        region_a = mask_a == label_a
        for j, label_b in enumerate(labels_b):
            region_b = mask_b == label_b
            intersection = np.count_nonzero(region_a & region_b)
            union = np.count_nonzero(region_a | region_b)
            iou[i, j] = intersection / union

    return iou

def link(mask_a, mask_b, iou_threshold=0.1):

    labels_a = [n for n in np.unique(mask_a) if n != 0]
    labels_b = [n for n in np.unique(mask_b) if n != 0]

    if not labels_a or not labels_b:
        return {}

    iou = iou_matrix(mask_a, mask_b, labels_a, labels_b)

    rows, cols = linear_sum_assignment(iou, maximize=True)

    return {int(labels_a[i]): int(labels_b[j]) for i, j in zip(rows, cols) if iou[i, j] >= iou_threshold}
