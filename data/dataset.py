"""
General class for LFCT dataset. Holds images, ground-truth masks and prediction masks. 

Expected format is:
    full frames:
        01_PC - raw .tif images
        01_GT - ground truth man_track.txt/xml and .tif masks
        01_PRED - prediction man_track.txt/xml and .tif masks
    every 2nd frame
        02_PC - raw .tif images
        02_GT - ground truth man_track.txt/xml and .tif masks
        02_PRED - prediction man_track.txt/xml and .tif masks
    etc.
"""

import tifffile
from pathlib import Path
import numpy as np
import os.path
import xml.etree.ElementTree as ET

# Class holds images and directories
class Dataset:
    def __init__(self, img_path=None, gt_path=None, pred_path=None, seg_path=None):
        self.img_dir = Path(img_path).expanduser() if img_path else None
        self.imgs = self._load_imgs(self.img_dir) if img_path else None

        self.gt_dir = Path(gt_path).expanduser() if gt_path else None
        self.pred_dir = Path(pred_path).expanduser() if pred_path else None
        self.seg_dir = Path(seg_path).expanduser() if seg_path else None

        self.gt_graph = self._load_graph_ctc(self.gt_dir / "man_track.txt") if gt_path else None
        self.gt_masks = self._load_masks(self.gt_dir) if gt_path else None

        self.pred_graph = self._load_graph_ctc(self.pred_dir / "man_track.txt") if pred_path else None
        self.pred_masks = self._load_masks(self.pred_dir) if pred_path else None

        self.masks_seg = self._load_masks(self.seg_dir) if seg_path else None

    def _load_imgs(self, in_dir):
        in_files = sorted(in_dir.glob("*.tif"))
        data = np.stack([tifffile.imread(f) for f in in_files])

        return data

    def _load_masks(self, in_dir):
        in_files = sorted(in_dir.glob("*.tif"))

        if in_files:
            return np.stack([tifffile.imread(f) for f in in_files])

        return None

    def _load_graph_ctc(self, in_dir):
        if os.path.isfile(in_dir):
            return np.loadtxt(in_dir)

        return None

    def save_pred(self):
        for t, mask in enumerate(self.pred_masks):
            tifffile.imwrite(self.pred_dir / f"mask_{t:04d}.tif", mask.astype(np.uint16))

        np.savetxt(self.pred_dir / "man_track.txt", self.pred_graph, fmt="%d")

    def save_gt(self):
        for t, mask in enumerate(self.gt_masks):
            tifffile.imwrite(self.gt_dir / f"mask_{t:04d}.tif", mask.astype(np.uint16))

        np.savetxt(self.gt_dir / "man_track.txt", self.gt_graph, fmt="%d")

    def save_imgs(self):
        for t, img in enumerate(self.imgs):
            tifffile.imwrite(self.img_dir / f"pc_{t:04d}.tif", img)

    def save_seg(self):
        for t, img in enumerate(self.masks_seg):
            tifffile.imwrite(self.seg_dir / f"seg_{t:04d}.tif", img)