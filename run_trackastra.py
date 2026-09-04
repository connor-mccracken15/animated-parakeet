import os
import torch
import numpy as np
import tifffile
import argparse
from pathlib import Path

from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc

def run_trackastra(in_path, dataset, out_path):

    img_dir  = Path(in_path).expanduser() / dataset / "02"
    mask_dir = Path(in_path).expanduser() / dataset / "02_GT" / "SEG"

    if not (img_dir.exists() and mask_dir.exists()):
        raise FileNotFoundError(f"Dataset folders could not be foun: {img_dir}")

    out_dir = Path(out_path).expanduser() / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(img_dir.glob("*.tif"))
    mask_files = sorted(mask_dir.glob("*.tif"))

    if not (img_files and mask_files):
        raise FileNotFoundError("Image or mask .tif files could not be found.")

    imgs  = np.stack([tifffile.imread(f) for f in img_files])
    masks = np.stack([tifffile.imread(f) for f in mask_files])

    model = Trackastra.from_pretrained("ctc")

    print("Starting tracking...")

    track_graph, masks_tracked = model.track(imgs, masks, mode="greedy")

    graph_to_ctc(track_graph, masks_tracked, outdir=out_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="Run Trackastra on a CTC dataset.")
    parser.add_argument("--in-path", type=str, required=True)
    parser.add_argument("--out-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)

    return parser.parse_args()

def main():
    args = parse_args()
    
    run_trackastra(in_path=args.in_path, out_path=args.out_path, dataset=args.dataset)
    
if __name__ == "__main__":
    main()