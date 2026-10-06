"""
Run Cellpose segmentation on all frames
"""

import numpy as np
from cellpose import models
import copy

def run_cellpose(dataset):
    model = models.CellposeModel(pretrained_model="cpsam_v2", gpu=False)
    masks = []
    for i, img in enumerate(dataset.rfp):
        print(f"Running frame {i}")
        masks.append(model.eval(img, flow_threshold=0.5, cellprob_threshold=0.0)[0])
    return np.stack(masks)