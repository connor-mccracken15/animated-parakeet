from pathlib import Path

from data.lfct_dataset import LFCT_Dataset
from data.view_dataset import view_dataset
from data.trackmate_to_ctc import trackmate_to_ctc
from data.create_lfr import create_lfr
from tracking.evaluate_dataset import evaluate_dataset

tests = ["HEK293", "MDA-MB-231", "MFC10A", "U87"]
frame_drops = [2, 4, 8, 16]

in_dir = Path("~/projects/dissertation/data/lfct").expanduser()


dataset = LFCT_Dataset(in_path = in_dir / "MFC10A", gt_path = in_dir / "MFC10A")

lfr_dataset = create_lfr(dataset, 8)

view_dataset(lfr_dataset)