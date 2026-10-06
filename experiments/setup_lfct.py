"""
Generate initial datasets - converts from TrackMate to CTC format and creates low frame rate (LFR) sets
"""

from pathlib import Path

from data.convert_lfct import create_ctc
from data.dataset import Dataset

TYPES = ["HEK293", "MDA-MB-231", "MFC10A", "U87"]
N_FRAMES = [2, 4, 8, 16, 32]

def main():
    in_path = Path("~/projects/dissertation/data/lfct").expanduser()

    for t in TYPES:

        dataset = Dataset(img_path = in_path / t / "01_PC",  gt_path = in_path / t / "01_GT", pred_path = in_path / t / "01_PRED")

        dataset.gt_masks, dataset.gt_graph = create_ctc(dataset.imgs, in_path / t / "01_GT" / "man_track.xml")
        dataset.save_gt()

if __name__ == "__main__":
    main()
