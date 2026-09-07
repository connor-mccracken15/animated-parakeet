import numpy as np
import tifffile
import argparse
from pathlib import Path

from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc

def run_trackastra(img_path, mask_path, out_path, mode):

    img_dir  = Path(img_path).expanduser()
    mask_dir = Path(mask_path).expanduser()

    if not (img_dir.exists() and mask_dir.exists()):
        raise FileNotFoundError(f"Dataset folders could not be foun: {img_dir}")

    out_dir = Path(out_path).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(img_dir.glob("*.tif"))
    mask_files = sorted(mask_dir.glob("*.tif"))

    if not (img_files and mask_files):
        raise FileNotFoundError("Image or mask .tif files could not be found.")

    imgs  = np.stack([tifffile.imread(f) for f in img_files])
    masks = np.stack([tifffile.imread(f) for f in mask_files])

    model = Trackastra.from_pretrained("ctc")

    print("Starting tracking...")

    track_graph, masks_tracked = model.track(imgs, masks, mode)

    graph_to_ctc(track_graph, masks_tracked, outdir=out_dir)