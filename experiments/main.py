from pathlib import Path

from data.lfct_dataset import LFCT_Dataset
from data.view_dataset import view_dataset
from data.trackmate_to_ctc import trackmate_to_ctc
from tracking.evaluate_dataset import evaluate_dataset

dataset = LFCT_Dataset(in_path="~/projects/dissertation/data/lfct/MFC10A", gt_path="~/projects/dissertation/data/lfct/MFC10A", pred_path="~/projects/dissertation/data/lfct/MFC10A")

results, matched = evaluate_dataset(dataset)