import os
import torch
import numpy as np

from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc, graph_to_napari_tracks, write_to_geff
from trackastra.data import example_data_bacteria

def load_gpu(index):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)

    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs (visible):", torch.cuda.device_count())

    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA GPU at index {index} is not available.")

    # Run test - throws error if any issues
    try:
        torch.randn(1000, 1000, device=torch.device("cuda"))
    except:
        print("Unable to run tensor test - check GPU.")

    return "cuda"

def run_trackastra_test(device):
    model = Trackastra.from_pretrained("general_2d", device=device)

    imgs, masks = example_data_bacteria()

    track_graph, masks_tracked = model.track(imgs, masks, mode="greedy")

    ctc_tracks, ctc_masks = graph_to_ctc(
        track_graph,
        masks_tracked,
        outdir="/scratch/output",
    )

def main():
    device = load_gpu(index=0)
    run_trackastra_test(device)

if __name__ == "__main__":
    main()