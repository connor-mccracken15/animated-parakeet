"""
euclidian Distance + Hungarian linking between two label masks.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from scipy.spatial.distance import cdist
from skimage.measure import regionprops

def _centroids(mask, labels):
    c = {r.label: r.centroid for r in regionprops(mask)}
    return np.array([c[l] for l in labels]).reshape(-1, 2)

def _distance_matrix(mask_a, mask_b, labels_a, labels_b):
    return cdist(_centroids(mask_a, labels_a), _centroids(mask_b, labels_b))

def _link(mask_a, mask_b, max_dist):

    labels_a = [n for n in np.unique(mask_a) if n != 0]
    labels_b = [n for n in np.unique(mask_b) if n != 0]

    if not labels_a or not labels_b:
        return {}

    dist = _distance_matrix(mask_a, mask_b, labels_a, labels_b)
    dist[dist > max_dist] = 1e6  # forbid links beyond max_dist

    rows, cols = linear_sum_assignment(dist)

    return {int(labels_a[i]): int(labels_b[j]) for i, j in zip(rows, cols) if dist[i, j] <= max_dist}

def run_distance_hungarian(dataset, max_dist=20):
    masks = dataset.gt_masks
    pred = np.zeros_like(masks)
    tracks = {}
    prev_map = {}
    next_id = 1

    for t in range(masks.shape[0]):
        links = _link(masks[t - 1], masks[t], max_dist) if t > 0 else {}
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

    pred_graph = np.array([[tid, b, e, 0] for tid, (b, e) in tracks.items()])

    return pred, pred_graph 


