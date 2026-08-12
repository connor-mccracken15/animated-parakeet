import os
import torch
import numpy as np
import tifffile

from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc

def load_gpu(index):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)

    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs (visible):", torch.cuda.device_count())

    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA GPU at index {index} is not available.")

    return "cuda"

def run_trackastra_test(device, input_dir, dataset, output_dir):    

    img_dir = os.path(dataset, input_dir, "01")
    mask_dir = os.path(dataset, input_dir, "01_GT", "SEG")

    if not (os.path.exists(img_dir) and os.path.exists(mask_dir)):
        raise RuntimeError("Datasets could not be found.")

    output_dir = os.path(output_dir, dataset)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    imgs  = np.stack([tifffile.imread(f) for f in sorted(img_dir.glob("*.tif"))])
    masks = np.stack([tifffile.imread(f) for f in sorted(mask_dir.glob("*.tif"))])

    model = Trackastra.from_pretrained("ctc", device=device)

    track_graph, masks_tracked = model.track(imgs, masks, mode="greedy")

    ctc_tracks, ctc_masks = graph_to_ctc(track_graph, masks_tracked, outdir=output_dir)

def main():
    device = load_gpu(index=0)

    run_trackastra_test(device)

if __name__ == "__main__":
    main()