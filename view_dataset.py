import argparse
import tifffile
import numpy as np
from pathlib import Path
import napari

def run_napari(img_path, gt_path=None, pred_path=None):

    viewer = napari.Viewer()

    img_dir = Path(img_path).expanduser()

    img_files = sorted(img_dir.glob("*.tif"))
    img_stack = np.stack([tifffile.imread(f) for f in img_files])

    viewer.add_image(img_stack, name="video")

    if gt_path:
        gt_dir = Path(gt_path).expanduser()
        gt_files = sorted(gt_dir.glob("*.tif"))
        gt_stack = np.stack([tifffile.imread(f) for f in gt_files])

        viewer.add_labels(gt_stack, name="gt masks")

    if pred_path:
        pred_dir = Path(pred_path).expanduser()
        pred_files = sorted(pred_dir.glob("*.tif"))
        pred_stack = np.stack([tifffile.imread(f) for f in pred_files])

        viewer.add_labels(pred_stack, name="pred masks")

    napari.run()
    