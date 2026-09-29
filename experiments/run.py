"""
Runs an experiment. Currently setup for testing only
"""

from pathlib import Path

from data.dataset import LFCT_Dataset
from visualisation.napari_view import view_dataset
from data.drop_framerate import create_lfr

from evaluation.metrics import evaluate_dataset
from tracking.iou_hungarian import run_iou_hungarian

in_dir = Path("~/projects/dissertation/data/lfct").expanduser()

dataset = LFCT_Dataset(in_path = in_dir / "MFC10A", gt_path = in_dir / "MFC10A")

lft_dataset = create_lfr(dataset, 16)

lft_dataset = run_iou_hungarian(lft_dataset)

lft_dataset.save_pred(in_dir / "MFC10A")

evaluate_dataset(lft_dataset)

view_dataset(lft_dataset)