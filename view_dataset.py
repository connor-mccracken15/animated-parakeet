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

def parse_args():
    parser = argparse.ArgumentParser(description="View tracking dataset.")
    parser.add_argument("--img-path"), 
    parser.add_argument("--gt-path", type=str, required=False),
    parser.add_argument("--pred-path", type=str, required=False)

    return parser.parse_args()

def main():
    args = parse_args()

    run_napari(img_path=args.img_path, gt_path=args.gt_path, pred_path=args.pred_path)

if __name__ == "__main__":
    main()
    