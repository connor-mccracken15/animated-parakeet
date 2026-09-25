"""
General class for LFCT dataset. Holds images, ground-truth masks and prediction masks. 
"""

import tifffile
from pathlib import Path
import numpy as np

from traccuracy.loaders import load_ctc_data

class LFCT_Dataset:
    def __init__(self, in_path=None, gt_path=None, pred_path=None):
        self.imgs = self._load_imgs(Path(in_path, "01_PC").expanduser()) if in_path else None

        self.gt_dir = Path(gt_path, "01_GT", "TRA").expanduser() if gt_path else None
        self.pred_dir = Path(pred_path, "01_PRED", "TRA").expanduser() if pred_path else None

        self.gt_graph = self._load_graph(self.gt_dir / "track_ctc.txt") if gt_path else None
        self.gt_masks = self._load_masks(self.gt_dir) if gt_path else None

        self.pred_graph = self._load_graph(self.pred_dir / "track_ctc.txt") if pred_path else None
        self.pred_masks = self._load_masks(self.pred_dir) if pred_path else None

    def to_traccuracy(self, which):
        if which == "gt":
            return load_ctc_data(str(self.gt_dir), str(self.gt_dir / "track_ctc.txt"), name="GT")
        if which == "pred":
            return load_ctc_data(str(self.pred_dir), str(self.pred_dir / "track_ctc.txt"), name="pred")

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
        np.savetxt(self.pred_dir / "track_ctc.txt", self.pred_graph, fmt="%d")