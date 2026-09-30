"""
Generate initial datasets - converts from TrackMate to CTC format and creates low frame rate (LFR) sets
"""

from pathlib import Path

from data.convert import create_ctc
from data.drop_framerate import create_lfr
from data.dataset import LFCT_Dataset

TYPES = ["HEK293", "MDA-MB-231", "MFC10A", "U87"]
N_FRAMES = [2, 4, 8, 16, 32]

def main():
    in_path = Path("~/projects/dissertation/data/lfct").expanduser()

    for t in TYPES:

        dataset = LFCT_Dataset(img_path = in_path / t / "01_PC",  gt_path = in_path / t / "01_GT", pred_path = in_path / t / "01_PRED")

        dataset = create_ctc(dataset)
        dataset.save_gt()

        for n in N_FRAMES:
            dropped_dataset = create_lfr(dataset, n)

            dir = (in_path / t / f"{n:02d}_PC")
            dir.mkdir(parents=True, exist_ok=True)
            dropped_dataset.img_dir = dir

            dir = (in_path / t / f"{n:02d}_GT")
            dir.mkdir(parents=True, exist_ok=True)
            dropped_dataset.gt_dir = dir

            dir = (in_path / t / f"{n:02d}_PRED")
            dir.mkdir(parents=True, exist_ok=True)
            dropped_dataset.pred_dir = dir

            dropped_dataset.save_gt()
            dropped_dataset.save_imgs()

if __name__ == "__main__":
    main()
