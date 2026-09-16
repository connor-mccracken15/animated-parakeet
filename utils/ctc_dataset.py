"""
General class for CTC dataset. Holds images, ground-truth masks and prediction masks. 
"""

import tifffile
from pathlib import Path
import numpy as np

class CTC_Dataset:
    def __init__(self, img_path, seg_path=None, trk_mask_gt_path=None, trk_mask_pred_path=None, trk_graph_gt_path=None, trk_graph_pred_path=None):

        self.img = self._load_imgs(img_path)

        self.seg = self._load_imgs(seg_path) if seg_path else None
        self.trk_mask_gt = self._load_imgs(trk_mask_gt_path) if trk_mask_gt_path else None
        self.trk_mask_pred = self._load_imgs(trk_mask_pred_path) if trk_mask_pred_path else None

        self.trk_graph_gt = self._load_graph(trk_graph_gt_path) if trk_graph_gt_path else None
        self.trk_graph_pred = self._load_graph(trk_graph_pred_path) if trk_graph_pred_path else None

    def _load_imgs(self, in_path):
        in_dir = Path(in_path).expanduser()
        in_files = sorted(in_dir.glob("*.tif"))
        data = np.stack([tifffile.imread(f) for f in in_files])

        return data

    def _load_masks(self, in_path):
        in_dir = Path(in_path).expanduser()
        in_files = sorted(in_dir.glob("*.tif"))
        data = np.stack([tifffile.imread(f) for f in in_files])

        return data

    def _load_graph(self, in_path):
        in_dir = Path(in_path).expanduser()
        data = np.loadtxt(in_dir)

        return data