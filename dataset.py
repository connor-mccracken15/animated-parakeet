"""
General class for dataset. Holds images, ground-truth masks and prediction masks. 
"""

import tifffile
from pathlib import Path
import numpy as np

class Dataset:
    def __init__(self, img_path, seg_path=None, trk_gt_path=None, trk_pred_path=None):
        self.k = 0
        self.img = self._load_data(img_path)
        self.seg = self._load_data(seg_path) if seg_path else None
        self.trk_gt = self._load_data(trk_gt_path) if trk_gt_path else None
        self.trk_pred = self._load_data(trk_pred_path) if trk_pred_path else None

    def _load_data(self, in_path):
        in_dir = Path(in_path).expanduser()
        in_files = sorted(in_dir.glob("*.tif"))
        data = np.stack([tifffile.imread(f) for f in in_files])

        return data

    def get_k_frames(self, type):
        frames = getattr(self, type)[0:self.k]
        return(frames)