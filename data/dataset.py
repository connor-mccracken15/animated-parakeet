"""
General class for LFCT dataset. Holds images, ground-truth masks and prediction masks. 
"""

import tifffile
from pathlib import Path
import numpy as np

# Class holds images and directories
class LFCT_Dataset:
    def __init__(self, img_path=None, gt_path=None, pred_path=None):
        self.imgs = self._load_imgs(Path(img_path).expanduser()) if img_path else None

        self.gt_dir = Path(gt_path).expanduser() if gt_path else None
        self.pred_dir = Path(pred_path).expanduser() if pred_path else None

        self.gt_graph = self._load_graph(self.gt_dir / "man_track.txt") if gt_path else None
        self.gt_masks = self._load_masks(self.gt_dir) if gt_path else None

        self.pred_graph = self._load_graph(self.pred_dir / "track_ctc.txt") if pred_path else None
        self.pred_masks = self._load_masks(self.pred_dir) if pred_path else None

    def _load_imgs(self, in_dir):
        in_files = sorted(in_dir.glob("*.tif"))
        data = np.stack([tifffile.imread(f) for f in in_files])

        return data

    def _load_masks(self, in_dir):
        in_files = sorted(in_dir.glob("*.tif"))
        data = np.stack([tifffile.imread(f) for f in in_files])

        return data

    def _load_graph(self, in_dir):
        data = np.loadtxt(in_dir)

        return data

    def save_pred(self, pred_path):
        self.pred_dir = Path(pred_path, "01_PRED", "TRA").expanduser()
        self.pred_dir.mkdir(parents=True, exist_ok=True)

        for t, mask in enumerate(self.pred_masks):
            tifffile.imwrite(self.pred_dir / f"mask{t:03d}.tif", mask.astype(np.uint16))

        np.savetxt(self.pred_dir / "man_track.txt", self.pred_graph, fmt="%d")