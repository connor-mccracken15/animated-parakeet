"""
IoU + Hungarian linking between two label masks.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

def _iou_matrix(mask_a, mask_b, labels_a, labels_b):

    iou = np.zeros((len(labels_a), len(labels_b)))

    for i, label_a in enumerate(labels_a):
        region_a = mask_a == label_a
        for j, label_b in enumerate(labels_b):
            region_b = mask_b == label_b
            intersection = np.count_nonzero(region_a & region_b)
            union = np.count_nonzero(region_a | region_b)
            iou[i, j] = intersection / union

    return iou

def _link(mask_a, mask_b, iou_threshold):

    labels_a = [n for n in np.unique(mask_a) if n != 0]
    labels_b = [n for n in np.unique(mask_b) if n != 0]

    if not labels_a or not labels_b:
        return {}

    iou = _iou_matrix(mask_a, mask_b, labels_a, labels_b)

    rows, cols = linear_sum_assignment(iou, maximize=True)

    return {int(labels_a[i]): int(labels_b[j]) for i, j in zip(rows, cols) if iou[i, j] >= iou_threshold}

def run_iou_hungarian(dataset, iou_threshold=0.1):
    masks = dataset.gt_masks
    pred = np.zeros_like(masks)
    tracks = {}
    prev_map = {}
    next_id = 1

    for t in range(masks.shape[0]):
        links = _link(masks[t - 1], masks[t], iou_threshold) if t > 0 else {}
        back = {b: a for a, b in links.items()}
        cur_map = {}

        for label in np.unique(masks[t]):
            if label == 0:
                continue
            if label in back:
                tid = prev_map[back[label]]
                tracks[tid][1] = t
            else:
                tid = next_id
                next_id += 1
                tracks[tid] = [t, t]
            cur_map[label] = tid
            pred[t][masks[t] == label] = tid

        prev_map = cur_map

    dataset.pred_masks = pred
    dataset.pred_graph = np.array([[tid, b, e, 0] for tid, (b, e) in tracks.items()])

    return dataset

