"""
General class for dataset. Holds images, ground-truth masks and prediction masks. 
To simulate online, current frame k tracked, and frames 0:k can be accessed.
todo: load_data and add_pred safety checks
"""

import tifffile
from pathlib import Path
import numpy as np

class Dataset:
    def __init__(self, img_path, gt_path=None, pred_path=None):
        self.k = 0
        self.img = self._load_data(img_path)
        if gt_path:
            self.gt = self._load_data(gt_path)
        if pred_path:
            self.pred = self._load_data(pred_path)

    def _load_data(self, in_path):
        in_dir = Path(in_path).expanduser()
        in_files = sorted(in_dir.glob("*.tif"))
        data = np.stack([tifffile.imread(f) for f in in_files])

        return data

    def add_pred(self, k, mask):
        self.pred[k] = mask

    def get_k_frames(self, type):
        frames = getattr(self, type)[0:self.k]
        print(frames)
        return(frames)