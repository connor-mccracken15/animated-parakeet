"""
Run Cellpose segmentation on all frames
"""

import numpy as np
import torch
from cellpose import models
from tqdm import tqdm

from skimage.segmentation import watershed

def run_cellpose(ims, markers, prob_thresh=0.0):
    model = models.CellposeModel(
        pretrained_model="cyto3",
        gpu=True,
        device=torch.device("cuda:1"),
    )

    masks = []
    for img, mk in zip(tqdm(ims, desc="Cellpose"), markers):
        _, flows, _ = model.eval(img, flow_threshold=0.5, cellprob_threshold=0.0)
        cellprob = flows[2]
        fg = (cellprob > prob_thresh) | (mk > 0)
        masks.append(watershed(-cellprob, mk, mask=fg))
    return np.stack(masks)