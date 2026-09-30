"""
Run Cellpose segmentation on all frames
"""

import numpy as np
from cellpose import models
import copy

def run_cellpose(dataset):
    seg_dataset = copy.deepcopy(dataset)
    model = models.CellposeModel(pretrained_model="cpsam_v2", gpu=False)

    res = np.zeros_like(dataset.gt_masks)

    for i, img in enumerate(dataset.imgs):
        print(f"Running frame {i}")
        mask = model.eval(img, flow_threshold=0.5, cellprob_threshold=0.0)[0]
        res[i] = mask

    seg_dataset.masks_seg = res

    return seg_dataset