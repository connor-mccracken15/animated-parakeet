import os
import torch
import numpy as np
import tifffile
import argparse
from pathlib import Path

from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc

def check_gpu(index):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)

    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs (visible):", torch.cuda.device_count())

    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA GPU at index {index} is not available.")

    return "cuda"

def run_trackastra(device, in_path, dataset, out_path):

    img_dir  = Path(in_path).expanduser() / dataset / "01"
    mask_dir = Path(in_path).expanduser() / dataset / "01_ST" / "SEG"

    if not (img_dir.exists() and mask_dir.exists()):
        raise FileNotFoundError("Dataset folders could not be found.")

    out_dir = Path(out_path).expanduser() / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(img_dir.glob("*.tif"))
    mask_files = sorted(mask_dir.glob("*.tif"))

    if not (img_files and mask_files):
        raise FileNotFoundError("Image or mask .tif files could not be found.")

    imgs  = np.stack([tifffile.imread(f) for f in img_files])
    masks = np.stack([tifffile.imread(f) for f in mask_files])

    model = Trackastra.from_pretrained("ctc", device=device)

    print("Starting tracking...")

    track_graph, masks_tracked = model.track(imgs, masks, mode="greedy")

    graph_to_ctc(track_graph, masks_tracked, outdir=out_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="Run Trackastra on a CTC dataset.")
    parser.add_argument("--in-path", type=str, required=True)
    parser.add_argument("--out-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)

    return parser.parse_args()

def main():
    args = parse_args()

    device = check_gpu(index=args.gpu_index)

    run_trackastra(device=device, in_path=args.in_path, out_path=args.out_path, dataset=args.dataset)
    
if __name__ == "__main__":
    main()