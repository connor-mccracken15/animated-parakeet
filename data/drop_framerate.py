"""
Creates new dataset with dropped frames, to simulate low frame rate. Also updates graph.
"""

from data.dataset import LFCT_Dataset
import numpy as np
import copy

def _drop_frames(dataset, every_n):
    
    return dataset.imgs[::every_n], dataset.gt_masks[::every_n]

def _drop_graph(graph, n_frames, keep):
    new_idx = np.full(n_frames, -1)
    new_idx[keep] = np.arange(len(keep))

    rows = []
    for L, B, E, P in graph.astype(int):
        frames = keep[(keep >= B) & (keep <= E)]
        if frames.size:
            rows.append([L, new_idx[frames[0]], new_idx[frames[-1]], P])

    rows = np.array(rows)
    labels = rows[:, 0]
    parents = rows[:, 3]
    parent_missing = ~np.isin(parents, labels)
    rows[parent_missing, 3] = 0
    
    return rows

def create_lfr(dataset, every_n):
    lfr_dataset = copy.deepcopy(dataset)

    n_frames = dataset.imgs.shape[0]
    keep = np.arange(0, n_frames, every_n)

    lfr_dataset.imgs, lfr_dataset.gt_masks = _drop_frames(dataset, every_n)
    lfr_dataset.gt_graph = _drop_graph(dataset.gt_graph, n_frames, keep)

    return lfr_dataset