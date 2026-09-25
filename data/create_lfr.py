"""
Creates new dataset with dropped frames, to simulate low frame rate. Also updates graph.
"""

from data.lfct_dataset import LFCT_Dataset
import numpy as np

def _drop_frames(full_dataset, every_n):
    
    return full_dataset.imgs[::every_n], full_dataset.gt_masks[::every_n]

def _drop_graph(graph, n_frames, keep):
    new_idx = np.full(n_frames, -1)
    new_idx[keep] = np.arange(len(keep))

    rows = []
    for L, B, E, P in graph.astype(int):
        frames = keep[(keep >= B) & (keep <= E)]
        if frames.size:
            rows.append([L, new_idx[frames[0]], new_idx[frames[-1]], P])

    rows = np.array(rows)
    rows[~np.isin(rows[:, 3], rows[:, 0]), 3] = 0
    return rows

def create_lfr(dataset, every_n):
    lfr_dataset = LFCT_Dataset()
    n_frames = dataset.imgs.shape[0]
    keep = np.arange(0, n_frames, every_n)

    lfr_dataset.imgs, lfr_dataset.gt_masks = _drop_frames(dataset, every_n)
    lfr_dataset.gt_graph = _drop_graph(dataset.gt_graph, n_frames, keep)

    return lfr_dataset