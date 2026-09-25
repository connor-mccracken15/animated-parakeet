from pathlib import Path

from data.lfct_dataset import LFCT_Dataset
from data.view_dataset import view_dataset
from data.create_lfr import create_lfr

from tracking.evaluate_dataset import evaluate_dataset
from tracking.iou_hungarian import run_iou_hungarian

tests = ["HEK293", "MDA-MB-231", "MFC10A", "U87"]
frame_drops = [2, 4, 8, 16]

in_dir = Path("~/projects/dissertation/data/lfct").expanduser()

dataset = LFCT_Dataset(in_path = in_dir / "MFC10A", gt_path = in_dir / "MFC10A")

lft_dataset = create_lfr(dataset, 16)

lft_dataset = run_iou_hungarian(lft_dataset)

lft_dataset.save_pred(in_dir / "MFC10A")

evaluate_dataset(lft_dataset)

view_dataset(lft_dataset)