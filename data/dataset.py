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
    def __init__(self, pc_path=None, rfp_path=None, gt_path=None, pred_path=None, seg_path=None):
        self.pc_dir = Path(pc_path).expanduser() if pc_path else None
        self.pc = self._load_imgs(self.pc_dir) if pc_path else None

        self.rfp_dir = Path(rfp_path).expanduser() if rfp_path else None
        self.rfp = self._load_imgs(self.rfp_dir) if rfp_path else None

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
        self.pred_dir.mkdir(parents=True, exist_ok=True)
        for t, mask in enumerate(self.pred_masks):
            tifffile.imwrite(self.pred_dir / f"mask_{t:04d}.tif", mask.astype(np.uint16))

        np.savetxt(self.pred_dir / "man_track.txt", self.pred_graph, fmt="%d")

    def save_gt(self):
        self.gt_dir.mkdir(parents=True, exist_ok=True)
        for t, mask in enumerate(self.gt_masks):
            tifffile.imwrite(self.gt_dir / f"mask_{t:04d}.tif", mask.astype(np.uint16))

        np.savetxt(self.gt_dir / "man_track.txt", self.gt_graph, fmt="%d")

    def save_pc(self):
        self.pc_dir.mkdir(parents=True, exist_ok=True)
        for t, img in enumerate(self.pc):
            tifffile.imwrite(self.pc_dir / f"pc_{t:04d}.tif", img)

    def save_rfp(self):
        self.rfp_dir.mkdir(parents=True, exist_ok=True)
        for t, img in enumerate(self.rfp):
            tifffile.imwrite(self.rfp_dir / f"rfp_{t:04d}.tif", img)

    def save_seg(self):
        self.seg_dir_dir.mkdir(parents=True, exist_ok=True)
        for t, img in enumerate(self.masks_seg):
            tifffile.imwrite(self.seg_dir / f"seg_{t:04d}.tif", img)